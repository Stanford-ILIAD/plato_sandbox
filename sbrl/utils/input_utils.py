import enum
import os
import threading
import time
from collections import Hashable
from multiprocessing import Pipe
from multiprocessing.context import Process
from typing import List, Dict, Callable

import sharedmem

from sbrl.experiments import logger
from sbrl.utils.core_utils import CloudPickleWrapper
from sbrl.utils.python_utils import AttrDict, timeit, get_cls_param_instance, get_required, \
    get_with_default


def query_string_from_set(query: str, valid_set: List[str], lower=True):
    if lower:
        valid_set = [s.lower() for s in valid_set]

    def correct_cond(s):
        return s in valid_set

    return query_from_condition(query, correct_cond, lower=lower)


def query_from_condition(query: str, correct_condition: Callable[[str], bool], lower=True):
    prefix = "Q: "
    out = None
    while True:
        out = input(prefix + str(query))
        if lower:
            out = out.lower()

        if correct_condition(out):
            break
        else:
            prefix = "Invalid response, Q: "
    return out


# some conditions
def is_float_condition(s: str) -> bool:
    try:
        f = float(s)
        return True
    except ValueError:
        return False


def is_int_condition(s: str) -> bool:
    try:
        i = int(s)
        return True
    except ValueError:
        return False


def get_str_from(args: List = (), dc: AttrDict = AttrDict()) -> str:
    s = ""
    for a in args:
        s = s + "%s, " % str(a)
    for k,v in dc.leaf_items():
        s = s + "%s: %s, " % (k, str(v))

    if len(s) > 1:
        s = s[:-2]  # skip last comma if it is there

    return s


class UniqueInput(Hashable):
    def __eq__(self, other):
        raise NotImplementedError

    def __hash__(self) -> int:
        raise NotImplementedError


"""
User input "interface" between python classes and an input interface. Runs its own thread/process.

register_callback: call a function when a user input was received
read_input: get the current state of all the registered callbacks

"""


class UserInput(object):
    def __init__(self, params: AttrDict, callbacks: Dict[UniqueInput, Callable]):
        self.callbacks = {}  # added to later
        self.callback_lock = threading.Lock()  # prevents data issues when adding new callbacks asynchronously
        self.all_triggers = []  # must be updated in register
        self._init_params_to_attrs(params)
        for trigger, fn in callbacks.items():
            self.register_callback(trigger, fn)

    def _init_params_to_attrs(self, params: AttrDict):
        self.display = get_with_default(params, "display", None)
        # # allow object initialization as well
        if self.display is not None and isinstance(self.display, AttrDict):
            self.display = get_cls_param_instance(self.display, "cls", "params", object)

    def bind_display(self, display):
        assert not self.has_display()
        self.display = display

    def has_display(self):
        return self.display is not None

    def get_display(self):
        return self.display

    def is_running(self):
        return self.running

    def has_new_inputs(self):
        raise NotImplementedError

    # don't override this, override the other
    def register_callback(self, trigger: UniqueInput, cb: Callable):
        self.callback_lock.acquire()
        self._register_callback(trigger, cb)
        self.callback_lock.release()

    def _register_callback(self, trigger: UniqueInput, cb: Callable):
        self.callbacks[trigger] = cb
        self.all_triggers.append(trigger)

    def is_registered(self, trigger: UniqueInput):
        return trigger in self.all_triggers

    def read_input(self, wait=False):
        raise NotImplementedError

    def end(self):
        self.running = False

    def run(self, once=False, user_callback=None, **kwargs):
        self.running = True
        while self.running:
            if once:
                self.running = False
            if user_callback is not None:
                user_callback(self)


class ProcessUserInput(UserInput):

    def _init_params_to_attrs(self, params: AttrDict):
        super()._init_params_to_attrs(params)
        self.base_ui_params = params.base_ui_params
        if self.has_display():
            self.base_ui_params.display = self.display  # if we were passed in a display, pass it to child too
        self.base_ui_cls = get_required(params, "base_ui_cls")
        self.shared_running = sharedmem.empty((1,), dtype=bool)

    def get_shared_run_state(self):
        return self.shared_running

    def is_running(self):
        return self.shared_running[0]

    # one way
    def call_cmd(self, cmd, *args, **kwargs):
        dc = AttrDict.from_dict(kwargs)
        dc.local_args = args
        self.base_ui_read_remote.send((cmd, dc))


    # def populate_display(self, *args, **kwargs):
    #     print("before")
    #     self.display.populate_display(*args, **kwargs)
    #     print("after")

    @staticmethod
    def base_ui_process(remote, parent_remote, pui_wrapper, shared_running, run_args):
        ## setup first
        pui = pui_wrapper.x
        parent_remote.close()
        logger.debug("BASE UI PROCESS beginning...")

        kwargs = run_args.as_dict()
        assert "user_callback" not in kwargs.keys(), "NOT SUPPORTED"

        # instantiate with existing callbacks
        ui_obj: UserInput = pui.base_ui_cls(pui.base_ui_params, pui.callbacks)
        ui_obj.parent_ui = pui  # new attribute within UserInput, we are setting it in case callbacks want access

        assert isinstance(ui_obj, UserInput)

        waiting_for_read = False
        waiting_for_read_lock = threading.Lock()

        def command_thread_fn():
            nonlocal waiting_for_read

            while shared_running[0]:
                cmd, data = remote.recv()  # data is an AttrDict
                # logger.debug("[BaseUI] Received -> %s, %s" % (cmd, data))  # for debugging purposes
                # execution instructions on commands, these all are requests coming in on a socket (fast handling)
                if cmd == "register_callback":
                    # NOTE: the callback must be stateless or modify shared memory
                    trigger, cb_wrapper = data.trigger, data.cb_wrapper
                    ui_obj.register_callback(trigger, cb_wrapper.x)
                elif cmd == "has_new_inputs":
                    remote.send(("has_new_inputs", ui_obj.has_new_inputs()))
                elif cmd == "read_input":
                    waiting_for_read_lock.acquire()
                    waiting_for_read = get_with_default(data, "wait", False)
                    waiting_for_read_lock.release()
                    if not waiting_for_read:
                        remote.send(("read_input", ui_obj.read_input(wait=False)))
                elif hasattr(ui_obj, cmd):
                    # generic one way fn
                    args = list(data.local_args)
                    kwargs = data.as_dict()
                    del kwargs['args']
                    getattr(ui_obj, cmd)(*args, **kwargs)
                else:
                    logger.warn("Ignoring unexpected command %s with data %s" % (cmd, data))

        def user_callback(_):
            nonlocal waiting_for_read
            if not shared_running[0]:
                # externally end the obj
                ui_obj.end()

            elif waiting_for_read and ui_obj.has_new_inputs():
                remote.send(("read_input", ui_obj.read_input(wait=False)))
                waiting_for_read_lock.acquire()
                waiting_for_read = False
                waiting_for_read_lock.release()

        cmd_thread = threading.Thread(target=command_thread_fn)
        cmd_thread.start()
        ui_obj.run(user_callback=user_callback, **kwargs)
        cmd_thread.join()

    def _register_callback(self, trigger: UniqueInput, cb: Callable):
        self.callbacks[trigger] = cb
        self.all_triggers.append(trigger)

        if hasattr(self, "base_ui_proc") and self.base_ui_proc.is_alive():
            # stateful callbacks TODO
            self.base_ui_read_remote.send(("register_callback", AttrDict(trigger=trigger, cb_wrapper=CloudPickleWrapper(cb))))

    def read_input(self, wait=False):
        # print("sending...")
        self.base_ui_read_remote.send(("read_input", AttrDict(wait=wait)))
        # wait for info back
        cmd, data = self.base_ui_read_remote.recv()
        assert cmd == "read_input"
        return data

    def has_new_inputs(self):
        self.base_ui_read_remote.send(("has_new_inputs", AttrDict()))
        cmd, data = self.base_ui_read_remote.recv()
        assert cmd == "has_new_inputs"
        return data

    def end(self):
        self.shared_running[0] = False
        self.base_ui_proc.join()

    # asynchronous
    def run(self, **kwargs):

        attrs_for_run = AttrDict.from_dict(kwargs)

        self.shared_running[0] = True

        self.base_ui_read_remote, self.base_ui_work_remote = Pipe()
        self.base_ui_proc = Process(target=ProcessUserInput.base_ui_process,
                                    args=(self.base_ui_work_remote, self.base_ui_read_remote, CloudPickleWrapper(self),
                                          self.shared_running, attrs_for_run))
        self.base_ui_proc.daemon = True  # if the main process crashes, we should not cause things to hang
        self.base_ui_proc.start()
        self.base_ui_work_remote.close()


# universal for all key inputs
class KeyInput(UniqueInput):
    class ON(enum.IntEnum):
        pressed = 1
        down = 2
        up = 3

    def __init__(self, key: str, on: ON):
        self.key = key
        self.on = on

    def __eq__(self, other):
        return self.key == other.key and self.on == other.on

    def __hash__(self) -> int:
        return hash(self.key + self.on.name)

    def __str__(self):
        return "KeyInput: %s, on: %s" % (self.key, self.on.name)


# NOTE these keys must be registered first!!
# async means do not also get events from pygame (e.g. run was called from separate thread)
def wait_for_keydown_from_set(input_handle, valid_set: List[KeyInput], do_async=True) -> KeyInput:
    out = None
    valid_set_str = []
    for trig in valid_set:
        valid_set_str.append(trig.key.lower())
        assert input_handle.is_registered(trig), "Trigger not handled: " + str(trig)
    while out is None:
        if not do_async:
            input_handle.run(dt=0.1, once=True)
        key_states = input_handle.read_input(wait=do_async)
        for key, on_states in key_states.items():
            if key in valid_set_str:
                for i in range(len(on_states)):
                    try:
                        idx = valid_set.index(KeyInput(key, on_states[i]))
                        out = valid_set[idx]
                        break
                    except ValueError as e:
                        pass
            if out is not None:
                break
    return out


if __name__ == '__main__':
    import cv2
    from sbrl.utils.pygame_utils import TextFillPygameDisplay, PygameOnlyKeysInput
    rs = sbrl.envs.sensor.camera.RSDepthCamera(AttrDict(config_json=os.path.expanduser("~/test_config.json")))
    rs.open()
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    # import matplotlib.pyplot as plt

    pui = ProcessUserInput(
        AttrDict(
            base_ui_cls=PygameOnlyKeysInput,
            base_ui_params=AttrDict(),
            display=AttrDict(
                cls=TextFillPygameDisplay,
                params=AttrDict(),
            ),
        ), {})

    KI = KeyInput
    def handler(ui, ki):
        print("key: %s" % ki)
    pui.register_callback(KI("y", KI.ON.down), handler)
    pui.register_callback(KI("n", KI.ON.down), handler)
    pui.register_callback(KI("q", KI.ON.down), handler)
    pui.register_callback(KI("r", KI.ON.down), handler)
    pui.run()

    i = 0
    while i < 500:
        with timeit("populate"):
            pui.call_cmd("populate_display", "hello", "press enter to continue", "variable: %d" % i)
        with timeit("sleep"):
            time.sleep(0.05)
        cv2.imshow("test", rs.read_state().rgb)
        cv2.waitKey(1)
        i += 1

    pui.run()

    # ### PYGAME
    # pygame.init()
    # pygame.font.init()
    # screen = pygame.display.set_mode((400, 300))
    # screen.fill((255, 255, 255))
    # pygame.display.set_caption("Demonstrations")
    # myfont = pygame.font.SysFont('Arial', 20)
    # textline1 = myfont.render('In window, press..', False, (0, 0, 0))
    # textline2 = myfont.render('r: reset', False, (0, 0, 0))
    # textline3 = myfont.render('q: quit', False, (0, 0, 0))
    # screen.blit(textline1, (10, 10))
    # screen.blit(textline2, (10, 50))
    # screen.blit(textline3, (10, 90))
    # pygame.display.flip()
    #
    # KI = KeyInput
    #
    # def handler(ui, ki):
    #     # print("Handler called for %s" % ki.key)
    #     pass
    #
    # running = True
    # def quit(*args):
    #     global running
    #     running = False
    #
    # exit_on_ctrl_c()
    #
    # input_handle = PygameOnlyKeysInput(AttrDict(), {})
    # input_handle.register_callback(KI("i", KI.ON.pressed), handler)
    # input_handle.register_callback(KI("l", KI.ON.down), handler)
    # input_handle.register_callback(KI("j", KI.ON.up), handler)
    # input_handle.register_callback(KI("k", KI.ON.up), handler)
    # input_handle.register_callback(KI("k", KI.ON.down), handler)
    #
    # input_handle.register_callback(KI("q", KI.ON.down), quit)
    #
    # bg_thread = threading.Thread(target=input_handle.run, daemon=True)
    # bg_thread.start()
    #
    # time.sleep(1.0)
    #
    # input_handle.register_callback(KI("m", KI.ON.down), handler)
    #
    # while running:
    #     print(input_handle.read_input())
    #     time.sleep(3.0)




