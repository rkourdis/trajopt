import pickle
from enum import Enum

class Guess(Enum):
    STANDING_V = 1

# TODO: THIS ISN'T REALLY AN INITIAL GUESS!
def load_initial_guess(pose_type: Guess):
    blobs = {
        Guess.STANDING_V: "standing_pose.bin"
    }

    with open(f"initial_guesses/{blobs[pose_type]}", "rb") as rf:
        guess = pickle.load(rf)

    return guess["q"], guess["v"], guess["tau"], guess["λ_local_wa"]