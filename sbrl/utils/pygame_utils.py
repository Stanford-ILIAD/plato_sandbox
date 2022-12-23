import os
import time
from typing import Callable

import pygame

from sbrl.experiments import logger
from sbrl.utils.input_utils import UserInput, UniqueInput, KeyInput
from sbrl.utils.python_utils import AttrDict, get_with_default, timeit, get_required

if 'DISPLAY' not in os.environ.keys():
    os.environ['SDL_VIDEODRIVER'] = "dummy"

class PygameDisplay(object):
    def __init__(self, params):
        self._init_params_to_attrs(params)

    def _init_params_to_attrs(self, params: AttrDict):
        self.W = get_with_default(params, "W", 400)
        self.H = get_with_default(params, "H", 300)
        self.title = get_with_default(params, "title", "Pygame Window")
        self.bg_color = get_with_default(params, "bg_color", (255, 255, 255), map_fn=tuple)
        self._created = False

    def is_created(self):
        return self._created

    def create_display(self):
        assert not pygame.get_init()
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.W, self.H))
        self.screen.fill(self.bg_color)
        self.main_font = pygame.font.SysFont('Arial', 20)  # parameterize later
        pygame.display.set_caption(self.title)
        pygame.display.flip()
        self._created = True

    def destroy_display(self):
        assert pygame.get_init()
        pygame.quit()
        logger.warn("Destroying Pygame Display...")
        self._created = False

    def populate_display(self, **kwargs):
        textline1 = self.main_font.render('In window, press..', False, (0, 0, 0))
        # textline_y = myfont.render('y: yes', False, (0, 0, 0))
        # textline_n = myfont.render('n: no', False, (0, 0, 0))
        textline_r = self.main_font.render('r: reset', False, (0, 0, 0))
        textline_q = self.main_font.render('q: quit', False, (0, 0, 0))
        self.screen.blit(textline1, (10, 10))
        # screen.blit(textline_y, (10,50))
        # screen.blit(textline_n, (10,90))
        self.screen.blit(textline_r, (10, 50))
        self.screen.blit(textline_q, (10, 90))
        pygame.display.flip()  # end all populates with this


# takes in an already initialized display, ignores most of the params.
class PygameDisplayWrapper(PygameDisplay):

    def _init_params_to_attrs(self, params: AttrDict):
        super(PygameDisplayWrapper, self)._init_params_to_attrs(params)
        self._create_display_fn = get_required(params, "create_display_fn")  # returns a screen
        self._populate_display_fn = get_with_default(params, "populate_display_fn", lambda screen, **kwargs: None)  # populates a display

    def create_display(self):
        pygame.font.init()
        self.screen = self._create_display_fn()
        pygame.display.flip()
        self._created = True

    def populate_display(self, **kwargs):
        self._populate_display_fn(self.screen, **kwargs)
        pygame.display.flip()


# super basic
class TextFillPygameDisplay(PygameDisplay):

    def _init_params_to_attrs(self, params: AttrDict):
        super(TextFillPygameDisplay, self)._init_params_to_attrs(params)
        self.line_sep = get_with_default(params, "line_sep", 40)

    def populate_display(self, *text_lines: str, **kwargs):
        self.screen.fill(self.bg_color)  # TODO
        shift_x = 10
        shift_y = 10
        for text in text_lines:
            t = self.main_font.render(text, False, (0, 0, 0))
            self.screen.blit(t, (shift_x, shift_y))
            shift_y += self.line_sep
        pygame.display.flip()


if __name__ == '__main__':
    pg = TextFillPygameDisplay(AttrDict())
    pg.create_display()

    i = 0
    while i < 50:
        with timeit("populate"):
            pg.populate_display("hello", "press enter to continue", "variable: %d" % i)
        with timeit("sleep"):
            time.sleep(0.1)

        i += 1

    print("Complete:\n", timeit)


class PygameOnlyKeysInput(UserInput):
    def _init_params_to_attrs(self, params: AttrDict):
        super()._init_params_to_attrs(params)
        #
        assert self.has_display() and isinstance(self.display, PygameDisplay)  # TODO
        if not self.display.is_created():
            logger.warn("[KI] Creating pygame display.")
            self.display.create_display()  # necessary, created only once!

        self.key_states = dict()  # str -> [ON]
        self.intermediate_key_states = dict()
        self.steps = 0

    def _register_callback(self, trigger: UniqueInput, cb: Callable):
        super()._register_callback(trigger, cb)
        assert isinstance(trigger, KeyInput), "Keys must be KeyInput type as of right now... " + str(trigger)
        self.key_states[trigger.key] = []  # empty means no events
        self.intermediate_key_states[trigger.key] = []
        pygame.key.key_code(trigger.key)  # throws a value error if the key is invalid

    def has_new_inputs(self):
        return len(self.intermediate_key_states.keys()) > 0 and any(len(item) > 0 for key, item in self.intermediate_key_states.items())

    def populate_display(self, *args, **kwargs):
        self.display.populate_display(*args, **kwargs)

    def run(self, dt=0.1, user_callback=None, rate_limited=True, once=False, debug=False):
        if rate_limited:
            clock = pygame.time.Clock()

        # logger.debug("PYGAME user input RUN beginning...")
        self.running = True
        while self.running:
            self.steps += 1

            # update key states
            with timeit("pygame_get"):
                keys = [pygame.key.key_code(tr.key) for tr in self.all_triggers]
                try:
                    # logger.debug("PYGAME test get")
                    events = pygame.event.get()
                    # logger.debug("PYGAME test get done")
                except pygame.error as e:
                    logger.warn("[UI] pygame ended:" + str(e))
                    self.running = False
                    break
            with timeit("pygame_event_processing"):
                self.callback_lock.acquire()
                for event in events:
                    if event.type == pygame.KEYDOWN and event.key in keys:
                        self.key_states[pygame.key.name(event.key)] = [KeyInput.ON.down, KeyInput.ON.pressed]
                    elif event.type == pygame.KEYUP and event.key in keys:
                        self.key_states[pygame.key.name(event.key)] = [KeyInput.ON.up]  # no longer pressed

                # call the callbacks
                for trigger in self.all_triggers:
                    if trigger.on in self.key_states[trigger.key]:
                        self.callbacks[trigger](self, trigger)  # call the trigger fn with 2 argument

                # this is where read "injects" itself (buffer of all events seen)
                for key, item in self.key_states.items():
                    self.intermediate_key_states[key].extend(item)
                    self.intermediate_key_states[key] = list(set(self.intermediate_key_states[key]))

                # update the key states by removing transient states (up and down)
                for key, on_list in self.key_states.items():
                    if KeyInput.ON.up in on_list:
                        on_list.remove(KeyInput.ON.up)
                    if KeyInput.ON.down in on_list:
                        on_list.remove(KeyInput.ON.down)

                self.callback_lock.release()

            if user_callback is not None:
                with timeit("user_callback"):
                    user_callback(self)

            if rate_limited:
                with timeit("tick"):
                    clock.tick(int(1/dt))

            if once:
                self.running = False  # breaks from loop
            
            if debug and self.steps % int(5/dt) == 0:
                logger.debug(str(timeit))
                timeit.reset()

    # async read the states of all the monitored keys
    def read_input(self, wait=False):
        if wait:
            start = self.steps
            while self.steps == start:
                pass
        self.callback_lock.acquire()
        ret = self.intermediate_key_states.copy()
        for key in self.intermediate_key_states.keys():
            self.intermediate_key_states[key] = []  # reset it
        self.callback_lock.release()
        return ret
