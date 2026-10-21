# -*- coding: utf-8 -*-
from psychopy import visual, core, event, sound
from pylsl import StreamInfo, StreamOutlet, local_clock
from PIL import Image, ImageSequence
import numpy as np
import random

# ========================
# FILE PATHS (SET THESE)
# ========================
LEFT_GIF  = r"D:\Term 8\NBIC\MIBCI\Emotiv Data Acquisition\leftblackBG_rotated.gif"
RIGHT_GIF = r"D:\Term 8\NBIC\MIBCI\Emotiv Data Acquisition\rightblackBG_rotated.gif"
# (Forward slashes also work on Windows.)

# ========================
# Experiment parameters
# ========================
nTrials = 54
trialDuration = 4.0
restDuration = 3.0
blockSize = 18
fixationDuration = 6.0
demonstrationDuration = 2.0
openCloseEyes = 60.0

# Size of GIFs relative to window height (e.g., 0.5 = 50% of window height)
GIF_SCALE = 0.4
# Playback rate: 1.0 = normal, <1 = slower, >1 = faster
PLAYBACK_RATE = 0.7   # half-speed (slower)

# Rotation angles (degrees)
LEFT_ANGLE  = +30   # left cue rotates +45°
RIGHT_ANGLE = -30   # right cue rotates -45°

# Horizontal positions (in pixels) relative to window center
# (negative = left side, positive = right side)
LEFT_X_OFFSET  = -320
RIGHT_X_OFFSET = +320
Y_OFFSET       = 0

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
        name='MotorImageryMarkers',
        type='Markers',
        channel_count=1,
        nominal_srate=0,
        channel_format='string',
        source_id='motor_imagery_exp'
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
win = visual.Window(size=[1080, 520], color="black", units="pix", fullscr=False)

restText = visual.TextStim(win, text="REST", color="white", height=40)
breakText = visual.TextStim(win, text="BREAK - Press ENTER to continue", color="white", height=30)
fixationCross = visual.TextStim(win, text="+", color="white", height=40)
eyeOpenMsg = visual.TextStim(win, text="Please OPEN your eyes", color="white", height=40)
eyeCloseMsg = visual.TextStim(win, text="Please CLOSE your eyes", color="white", height=40)
startMsg = visual.TextStim(win, text="Welcome to the Motor Imagery Task\n\nPress SPACE to begin", color="white", height=30)
openEyesSignal = sound.Sound(880, secs=1.0)

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

# ========================
# GIF loader / player
# ========================
def load_gif_as_arrays(path):
    """
    Load GIF frames as RGB float32 arrays scaled to [-1, 1] (PsychoPy format).
    """
    im = Image.open(path)
    frames, durations = [], []
    for fr in ImageSequence.Iterator(im):
        arr_u8 = np.asarray(fr.convert("RGB"), dtype=np.uint8)
        if not arr_u8.flags['C_CONTIGUOUS']:
            arr_u8 = np.ascontiguousarray(arr_u8)
        arr = (arr_u8.astype(np.float32) / 127.5) - 1.0  # -> [-1, 1]
        frames.append(arr)
        durations.append(fr.info.get("duration", 100) / 1000.0)  # ms -> s
    return frames, durations

def play_gif(win, frames, durations, seconds, *, scale=GIF_SCALE, angle=0, pos=(0, 0), playback_rate=1.0):
    """
    Play frames for `seconds`. You can set rotation via `angle` (deg) and
    on-screen position via `pos=(x,y)` in pixels.
    """
    win_w, win_h = win.size
    h0, w0 = frames[0].shape[:2]
    img_h = max(1, int(win_h * scale))
    img_w = max(1, int(img_h * (w0 / float(h0))))

    stim = visual.ImageStim(
        win,
        image=frames[0],
        size=(img_w, img_h),
        interpolate=True,
        colorSpace='rgb',   # expecting [-1,1]
        pos=pos,
        ori=angle           # rotation in degrees
    )
    # ... existing setup ...
    clock = core.Clock()
    idx = 0
    while clock.getTime() < seconds:
        abort_if_escape()
        stim.setImage(frames[idx])
        stim.draw()
        win.flip()

        # scale the per-frame wait by playback_rate
        remaining = durations[idx] / playback_rate
        while remaining > 0:
            abort_if_escape()
            step = min(0.01, remaining)
            core.wait(step)
            remaining -= step

        idx = (idx + 1) % len(frames)

# Preload GIFs
left_frames, left_durs = load_gif_as_arrays(LEFT_GIF)
right_frames, right_durs = load_gif_as_arrays(RIGHT_GIF)

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
# Demonstration phase (GIFs)
# ========================
win.callOnFlip(send_marker, 'DemoLeftStart')
play_gif(win, left_frames, left_durs, demonstrationDuration,
         scale=GIF_SCALE, angle=LEFT_ANGLE, pos=(LEFT_X_OFFSET, Y_OFFSET),
         playback_rate=PLAYBACK_RATE)


win.callOnFlip(send_marker, 'DemoRightStart')
play_gif(win, right_frames, right_durs, demonstrationDuration,
         scale=GIF_SCALE, angle=RIGHT_ANGLE, pos=(RIGHT_X_OFFSET, Y_OFFSET),
         playback_rate=PLAYBACK_RATE)



win.callOnFlip(send_marker, 'FixationStart')
timed_wait(fixationDuration, draw_stim=fixationCross)

# ========================
# Trial (GIF cues)
# ========================
def run_trial(condition, trial_num):
    if condition == "left":
        frames, durs = left_frames, left_durs
        angle, pos = LEFT_ANGLE, (LEFT_X_OFFSET, Y_OFFSET)
    else:
        frames, durs = right_frames, right_durs
        angle, pos = RIGHT_ANGLE, (RIGHT_X_OFFSET, Y_OFFSET)

    win.callOnFlip(send_marker, f'trial={trial_num};cond={condition};event=cue_on')
    play_gif(win, frames, durs, trialDuration, scale=GIF_SCALE, angle=angle, pos=pos,
         playback_rate=PLAYBACK_RATE)
    win.callOnFlip(send_marker, f'trial={trial_num};cond={condition};event=cue_end')

# ========================
# Main loop
# ========================
n_left = nTrials // 2
n_right = nTrials - n_left
conditions = ["left"] * n_left + ["right"] * n_right
rng.shuffle(conditions)

send_marker('MainTaskStart')

for i, cond in enumerate(conditions, start=1):
    abort_if_escape()
    run_trial(cond, i)

    # inter-trial fixation
    win.callOnFlip(send_marker, f'trial={i};cond={cond};event=imagine_on')
    timed_wait(fixationDuration, draw_stim=fixationCross)
    win.callOnFlip(send_marker, f'trial={i};cond={cond};event=imagine_end')

    # block break
    if i % blockSize == 0:
        win.callOnFlip(send_marker, 'RestStart')
        timed_wait(restDuration, draw_stim=restText)
        win.callOnFlip(send_marker, 'RestEnd')

        win.callOnFlip(send_marker, 'BreakStart')
        wait_for_keys(('return',), draw_stim=breakText)  # ENTER to continue
        win.callOnFlip(send_marker, 'BreakEnd')

# ========================
# Ends
# ========================
win.callOnFlip(send_marker, 'ExperimentEnd')
thanks = visual.TextStim(win, text="Thank you! Experiment finished.", color="white", height=40)
timed_wait(3, draw_stim=thanks)
win.close()
core.quit()

