# Import functions to ensure registration happens
from ces_tutorial.functions.router import router_fn
from ces_tutorial.functions.router_agent import router_agent_fn
from ces_tutorial.functions.reachy_look import reachy_look
from ces_tutorial.functions.reachy_see import reachy_see
from ces_tutorial.functions.reachy_count_people import reachy_count_people
from ces_tutorial.functions.reachy_look_around import reachy_look_around
from ces_tutorial.functions.reachy_gesture import reachy_gesture
from ces_tutorial.functions.reachy_emotion import reachy_emotion
from ces_tutorial.functions.reachy_face_tracking import reachy_face_tracking
from ces_tutorial.functions.web_search import web_search
from ces_tutorial.functions.remember_fact import remember_fact
from ces_tutorial.functions.recall_facts import recall_facts

__all__ = [
    "router_fn",
    "router_agent_fn",
    "reachy_look",
    "reachy_see",
    "reachy_count_people",
    "reachy_look_around",
    "reachy_gesture",
    "reachy_emotion",
    "reachy_face_tracking",
    "web_search",
    "remember_fact",
    "recall_facts",
]
