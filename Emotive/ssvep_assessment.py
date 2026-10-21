# -*- coding: utf-8 -*-
from psychopy import visual, core, event, sound
from psychopy.visual import MovieStim
from pylsl import StreamInfo, StreamOutlet, local_clock
from PIL import Image, ImageSequence
import numpy as np
import random

# ========================
# VIDEO FILES
# ========================
#r"D:\Term 8\NBIC\MIBCI\Emotiv Data Acquisition\summary\top_video.mp4"
TOP_VIDEO = r"D:\Term 8\NBIC\MIBCI\Emotiv Data Acquisition\summary\top_video.mp4"
BOTTOM_VIDEO = r"D:\Term 8\NBIC\MIBCI\Emotiv Data Acquisition\summary\bottom_video.mp4"


# ========================
# SSVEP Parameters
# ========================
'''
TOP_TRIALS = 2
BOTTOM_TRIALS = 2
VIDEO_DURATION = 3.0
openCloseEyes = 2.0
'''
TOP_TRIALS = 25
BOTTOM_TRIALS = 25
openCloseEyes = 60.0
VIDEO_DURATION = 6.0

FIXATION_DURATION = 2.0
INSTRUCTION_DURATION = 1.0
REST_DURATION = 2.0



# ========================
# Participant / seed
# ========================
participant_id = "001"
seed = int(participant_id)
rng = random.Random(seed)
print(f"Random seed used for trial order: {seed}")

# ========================
# LSL Marker Setup
# ========================
def setup_lsl_markers():
    info = StreamInfo(
        name='SSVEPMarkers',
        type='Markers',
        channel_count=1,
        nominal_srate=0,
        channel_format='string',
        source_id='ssvep_exp'
    )
    return StreamOutlet(info)

marker_outlet = setup_lsl_markers()

def send_marker(marker):
    ts = local_clock()
    marker_outlet.push_sample([marker], ts)
    print(f"Sent marker: {marker} at {ts}")

# ========================
# Window / static stimuli
# ========================
#
win = visual.Window(
    size=(1130,570),
    pos=(70, 50),      # manually center on 1920x1080 screen
    fullscr=False,
    allowGUI=True,
    screen=0,
    waitBlanking=True,
    color="black",
    units="pix"
)

restText = visual.TextStim(win, text="REST", color="white", height=40)
breakText = visual.TextStim(win, text="BREAK - Press ENTER to continue", color="white", height=30)
fixationCross = visual.TextStim(win, text="+", color="white", height=40)
eyeOpenMsg = visual.TextStim(win, text="Please OPEN your eyes", color="white", height=40)
eyeCloseMsg = visual.TextStim(win, text="Please CLOSE your eyes", color="white", height=40)
startMsg = visual.TextStim(win, text="Welcome to the SSVEP Assessment\n\nPress SPACE to begin", color="white", height=30)
openEyesSignal = sound.Sound(880, secs=1.0)

topInstruction = visual.TextStim(
    win,
    text="Focus on the TOP flicker",
    color="white",
    height=35
)

bottomInstruction = visual.TextStim(
    win,
    text="Focus on the BOTTOM flicker",
    color="white",
    height=35
)
choicePrompt = visual.TextStim(
    win,
    text="Choose LEFT or RIGHT\n\nPress ← or →",
    color="white",
    height=35
)

imageryPrompt = visual.TextStim(
    win,
    text="+\n\nImagine the chosen movement",
    color="white",
    height=35
)
# ========================
# Escape helpers
# ========================
def abort_if_escape():
    if 'escape' in event.getKeys():
        send_marker('ExperimentAborted')
        win.close()
        core.quit()

def wait_for_keys(allowed=('space',), draw_stim=None):
    clock = core.Clock()
    while True:
        abort_if_escape()
        if draw_stim is not None:
            draw_stim.draw()
            win.flip()
        keys = event.getKeys()
        if any(k in allowed for k in keys):
            return
        core.wait(0.01)

def timed_wait(seconds, draw_stim=None):
    clock = core.Clock()
    while clock.getTime() < seconds:
        abort_if_escape()
        if draw_stim is not None:
            draw_stim.draw()
            win.flip()
        core.wait(0.01)
def play_video(path):

    movie = MovieStim(
        win,
        filename=path,
        loop=False
    )

    win_w, win_h = win.size

    video_h = int(win_h * 0.90)
    video_w = int(video_h * 16 / 9)

    movie.size = (video_w, video_h)

    # center
    movie.pos = (0, 0)

    clock = core.Clock()

    while clock.getTime() < VIDEO_DURATION:

        abort_if_escape()

        movie.draw()
        win.flip()
# ========================
# Start screen
# ========================
startMsg.draw()
win.flip()
wait_for_keys(('space',), draw_stim=startMsg)
send_marker('ExperimentStart')

# ========================
# Eye-open / Eye-closed tasks
# ========================
win.callOnFlip(send_marker, 'EyesOpenStart')
timed_wait(openCloseEyes, draw_stim=eyeOpenMsg)

win.callOnFlip(send_marker, 'EyesClosedStart')
timed_wait(openCloseEyes, draw_stim=eyeCloseMsg)

openEyesSignal.play()
win.callOnFlip(send_marker, 'EyesSignal')
timed_wait(1.0)

# ========================
# SSVEP TASK
# ========================

conditions = (
    ["top"] * TOP_TRIALS +
    ["bottom"] * BOTTOM_TRIALS
)

rng.shuffle(conditions)

print("Trial order:", conditions)

send_marker("SSVEPTaskStart")

for trial_num, cond in enumerate(conditions, start=1):

    abort_if_escape()

    send_marker(
        f"trial={trial_num};event=fixation"
    )

    timed_wait(
        FIXATION_DURATION,
        draw_stim=fixationCross
    )

    if cond == "top":

        timed_wait(
            INSTRUCTION_DURATION,
            draw_stim=topInstruction
        )

        send_marker(
            f"trial={trial_num};target=top;freq=15;event=start"
        )

        play_video(
            TOP_VIDEO
        )

        send_marker(
            f"trial={trial_num};target=top;freq=15;event=end"
        )

    else:

        timed_wait(
            INSTRUCTION_DURATION,
            draw_stim=bottomInstruction
        )

        send_marker(
            f"trial={trial_num};target=bottom;freq=12;event=start"
        )

        play_video(
            BOTTOM_VIDEO
        )

        send_marker(
            f"trial={trial_num};target=bottom;freq=12;event=end"
        )

    timed_wait(
        REST_DURATION,
        draw_stim=restText
    )

send_marker("SSVEPTaskEnd")
# ========================
# Ends
# ========================
win.callOnFlip(send_marker, 'ExperimentEnd')
thanks = visual.TextStim(win, text="Thank you! Experiment finished.", color="white", height=40)
timed_wait(3, draw_stim=thanks)
win.close()
core.quit()

