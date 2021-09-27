#!/usr/bin/env python3
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import ffmpeg
import numpy as np
import cv2
from tqdm import tqdm, trange
from dv import AedatFile
import scipy.signal
import matplotlib.pyplot as plt

app = QApplication(sys.argv)

# VIDEO_FPS = 500
# VIDEO_FPS = 50
# VIDEO_FPS = 30
VIDEO_FPS = int(sys.argv[3])

class VideoViewer(QWidget):
    def __init__(self):
        super().__init__()

        probe = ffmpeg.probe(sys.argv[1])
        videoStream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        self.NFRAMES = int(videoStream['nb_frames'])
        self.WIDTH = int(videoStream['width'])
        self.HEIGHT = int(videoStream['height'])
        self.SCALE = min(640/self.WIDTH, 480/self.HEIGHT)*2

        self.inp = (
            ffmpeg
            .input(sys.argv[1], hwaccel='cuda')
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdout=True)
        )

        self.frames = []

        self.label = QLabel(self)
        self.label.setMouseTracking(True)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

    def preload(self, t):
        print(t)

        while len(self.frames) <= t*VIDEO_FPS:
            frame = self.inp.stdout.read(self.HEIGHT*self.WIDTH*3)
            if len(frame) == 0:
                break  # eof
            frame = np.frombuffer(frame, np.uint8).reshape([self.HEIGHT, self.WIDTH, 3])
            self.frames.append(frame)

    def update(self, t):
        self.preload(t)
        frame = self.frames[max(0, min(len(self.frames)-1, int(t*VIDEO_FPS)))]

        self.label.setPixmap(QPixmap.fromImage(QImage(
            frame.tobytes(), self.WIDTH, self.HEIGHT, 3*self.WIDTH,
            QImage.Format_RGB888)).scaled(self.WIDTH*self.SCALE, self.HEIGHT*self.SCALE, Qt.KeepAspectRatio))

def aedatlen(fn):
    res = 0
    aedat = AedatFile(sys.argv[2])
    aedatiter = iter(aedat['events'].numpy())

    evts = next(aedatiter)
    startts = evts['timestamp'][0]
    try:
        while True:
            evts = next(aedatiter)
    except StopIteration:
        return (evts['timestamp'][-1]-startts)/1000000

class EventViewer(QWidget):
    def __init__(self):
        super().__init__()

        self.WIDTH, self.HEIGHT = 240, 180
        self.SCALE = 4

        self.aedat = AedatFile(sys.argv[2])

        self.aedatiter = iter(self.aedat['events'].numpy())
        self.aedatlength = aedatlen(sys.argv[2])
        self.startts = None

        self.times = []
        self.events = []
        # self.ts = np.zeros((self.HEIGHT, self.WIDTH, 2))-100

        self.label = QLabel(self)
        self.label.setMouseTracking(True)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

        self.WS = 0.1

    def setWindowLength(self, ws):
        self.WS = ws

    def preload(self, t):
        lastts = self.times[-1] if len(self.times) > 0 else -1
        if self.startts is None:
            evts = next(self.aedatiter)
            self.startts = np.min(evts['timestamp'])
            lastts = (evts['timestamp']-self.startts)/1000000
            self.times = lastts
            self.events = evts

        try:
            tbuf = []
            ebuf = []
            maxtime = self.times[-1]
            while maxtime < t:
                evts = next(self.aedatiter)

                lastts = (evts['timestamp']-self.startts)/1000000

                if np.min(lastts%1)<self.times[-1]%1:
                    print(lastts[-1], 'out of', self.aedatlength)
                tbuf.append(lastts)
                ebuf.append(evts)
                maxtime = lastts[-1]

            self.times = np.hstack([self.times]+tbuf)
            self.events = np.hstack([self.events]+ebuf)
        except StopIteration:
            pass

    def update(self, t):
        self.preload(t)

        currentts = np.zeros((self.HEIGHT, self.WIDTH, 2))-100
        start = np.searchsorted(self.times, t-self.WS, 'right')
        end = np.searchsorted(self.times, t, 'left')

        ts = self.times[start:end+1]
        evts = self.events[start:end+1]
        xs, ys, ps = evts['x'], evts['y'], evts['polarity']
        currentts[ys, xs, 1-ps*1] = ts

        # self.ts = np.linspace(0, t, self.HEIGHT*self.WIDTH*2).reshape(self.HEIGHT, self.WIDTH, 2)

        # print(np.min(self.ts), np.max(self.ts))
        lastts = np.max(currentts)
        img = np.maximum((currentts-lastts+self.WS), 0)/self.WS
        # print(np.min(img), np.max(img))

        frame = np.zeros((self.HEIGHT, self.WIDTH, 3))
        frame[..., 0] = img[..., 0]
        frame[..., 2] = img[..., 1]
        frame = (frame*255).astype(np.uint8)

        self.label.setPixmap(QPixmap.fromImage(QImage(
            frame.tobytes(), self.WIDTH, self.HEIGHT, 3*self.WIDTH,
            QImage.Format_RGB888)).scaled(self.WIDTH*self.SCALE, self.HEIGHT*self.SCALE, Qt.KeepAspectRatio))

    def saveTrimmed(self, fn, tstart, tend):
        start = np.searchsorted(self.times, tstart, 'left')
        end = np.searchsorted(self.times, tend, 'right')

        times = self.times[start:end]
        events = self.events[start:end]

        np.savez_compressed(fn, times=times, events=events,
                            sourcev=sys.argv[1], sourcee=sys.argv[2],
                            tstart=tstart, tend=tend)
        print('saved')


class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('hfr-event syncer')
        self.videoViewer = VideoViewer()

        self.saveButton = QPushButton()
        self.saveButton.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.saveButton.setText('Save Trimmed Events...')
        self.saveButton.clicked.connect(self.save)

        self.autosyncButton = QPushButton()
        self.autosyncButton.setIcon(self.style().standardIcon(QStyle.SP_FileDialogStart))
        self.autosyncButton.setText('Auto-sync...')
        self.autosyncButton.clicked.connect(self.autoSync)

        self.playButton = QPushButton()
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        self.prevButton = QPushButton()
        self.prevButton.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekBackward))
        self.prevButton.clicked.connect(self.prev)

        self.nextButton = QPushButton()
        self.nextButton.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekForward))
        self.nextButton.clicked.connect(self.next)

        self.startButton = QPushButton()
        self.startButton.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipBackward))
        self.startButton.clicked.connect(self.seekStart)

        self.endButton = QPushButton()
        self.endButton.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipForward))
        self.endButton.clicked.connect(self.seekEnd)

        self.playbackRate = QComboBox()
        self.playbackRate.addItems(['5000', '2000', '1000', '500', '250', '120', '60', '30'])
        self.playbackRate.currentTextChanged.connect(self.playbackRateChanged)

        self.info = QLabel()

        self.isVideoPlaying = QCheckBox()
        self.isVideoPlaying.setChecked(True)

        widget = QWidget(self)
        self.setCentralWidget(widget)

        layout = QHBoxLayout()

        videoLayout = QVBoxLayout()
        videoLayout.addWidget(self.videoViewer)

        self.videoSeek = QSlider(Qt.Horizontal)
        self.videoSeek.setMinimum(0)
        self.videoSeek.setMaximum(self.videoViewer.NFRAMES*1000000//VIDEO_FPS)
        self.videoSeek.setSingleStep(1000)
        self.videoSeek.valueChanged.connect(self.seekVideo)
        videoLayout.addWidget(self.videoSeek)

        videoButtonLayout = QHBoxLayout()
        videoButtonLayout.addWidget(self.info)
        videoButtonLayout.addWidget(self.saveButton)
        videoButtonLayout.addWidget(self.autosyncButton)
        videoButtonLayout.addWidget(self.startButton)
        videoButtonLayout.addWidget(self.prevButton)
        videoButtonLayout.addWidget(self.playButton)
        videoButtonLayout.addWidget(self.nextButton)
        videoButtonLayout.addWidget(self.endButton)
        videoButtonLayout.addWidget(self.playbackRate)
        videoButtonLayout.addWidget(self.isVideoPlaying)
        videoLayout.addLayout(videoButtonLayout)

        eventLayout = QVBoxLayout()
        self.eventViewer = EventViewer()
        eventLayout.addWidget(self.eventViewer)

        self.eventSeek = QSlider(Qt.Horizontal)
        self.eventSeek.setMinimum(0)
        self.eventSeek.setMaximum(int(self.eventViewer.aedatlength*1000000))
        self.eventSeek.setSingleStep(1000)
        self.eventSeek.valueChanged.connect(self.seekEvent)
        eventLayout.addWidget(self.eventSeek)

        eventButtonLayout = QHBoxLayout()
        self.windowLength = QComboBox()
        self.windowLength.addItems(['0.1', '0.2', '0.5', '1', '2', '5', '10', '20', '50', '100', '200', '500'])
        self.windowLength.currentTextChanged.connect(self.windowLengthChanged)
        eventButtonLayout.addWidget(self.windowLength)

        eventLayout.addLayout(eventButtonLayout)

        layout.addLayout(videoLayout)
        layout.addLayout(eventLayout)

        widget.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.positionChanged)
        self.isPlaying = False
        self.t = 0.
        # self.eventOffset = 16.9078
        self.eventOffset = 0

        self.playbackRate.setCurrentText('500')
        self.windowLength.setCurrentText('10')
        self.updateViewers()

    def autoSync(self):
        print('preloading')
        self.videoViewer.preload(self.videoViewer.NFRAMES/VIDEO_FPS)
        self.eventViewer.preload(self.eventViewer.aedatlength)
        print('preloaded')

        print('computing video evt')
        frames = np.asarray(self.videoViewer.frames)[:, ::4, ::4].astype(np.float32)
        frames = np.sum(frames, 3)
        diff = np.abs(np.log(frames[1:]+1)-np.log(frames[:-1]+1))>1.5
        numVideoEvt = np.sum(diff.reshape(diff.shape[0], -1), 1)
        numVideoEvt = numVideoEvt/np.max(numVideoEvt)
        print('computed video evt')
        plt.plot(np.arange(len(numVideoEvt))/VIDEO_FPS, numVideoEvt, label='video')

        numEvt = np.zeros(int(self.eventViewer.aedatlength*VIDEO_FPS+1))
        arr = self.eventViewer.times
        for i in trange(numEvt.shape[0]):
            target = i/VIDEO_FPS
            pos = np.searchsorted(arr, target)
            numEvt[i] = pos
            arr = arr[pos:]
        numEvt = numEvt/np.max(numEvt)
        plt.plot(np.arange(len(numEvt))/VIDEO_FPS, numEvt, label='event')

        numVideoEvt = (numVideoEvt-numVideoEvt.mean())/numVideoEvt.std()
        numEvt = (numEvt-numEvt.mean())/numEvt.std()

        # correlation = scipy.signal.correlate(numVideoEvt, numEvt)
        # # mag = scipy.signal.correlate(np.ones_like(numVideoEvt), numEvt)
        # # mag = np.sqrt(scipy.signal.correlate(np.ones_like(numVideoEvt), numEvt**2))
        # mag = scipy.signal.correlate(np.ones_like(numVideoEvt), numEvt**2)
        # correlation /= mag

        def match(a, b):
            a = np.c_[a, a].astype(np.float32)
            b = np.c_[b, b].astype(np.float32)
            return cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED).reshape(-1)
            # return cv2.matchTemplate(a, b, cv2.TM_CCORR_NORMED).reshape(-1)
        correlation = match(numEvt, numVideoEvt)
        print(np.argmax(correlation))
        plt.plot(np.arange(len(correlation))/VIDEO_FPS, correlation, label='correlation')
        plt.legend()
        plt.show()
        best = np.where((correlation[1:-1]>correlation[:-2]) & (correlation[1:-1]>correlation[2:]))[0]
        print((best+1)/VIDEO_FPS)
        print(correlation[best+1])
        self.eventOffset = np.argmax(correlation[:-7000])/VIDEO_FPS
        self.updateViewers()


    def save(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setNameFilter("Events (*.npz)")
        dialog.setAcceptMode(QFileDialog.AcceptSave)

        newFn = sys.argv[1][:-4]+'.npz'
        dialog.selectFile(newFn)

        if dialog.exec_():
            newFn = dialog.selectedFiles()[0]
            tstart = self.eventOffset
            tend = self.eventOffset+self.videoViewer.NFRAMES/VIDEO_FPS
            self.eventViewer.preload(min(self.eventViewer.aedatlength, tend+1))
            self.eventViewer.saveTrimmed(newFn, tstart, tend)


    def windowLengthChanged(self, text):
        self.eventViewer.setWindowLength(float(text)/1000.)
        self.updateViewers()

    def playbackRateChanged(self, text):
        self.dt = 1./float(text)

        self.videoSeek.setSingleStep(1000000//float(text))
        self.eventSeek.setSingleStep(1000000//float(text))

    def seekVideo(self):
        delta = self.videoSeek.value()/1000000-self.t
        if self.isVideoPlaying.isChecked():
            self.t += delta
        else:
            self.t += delta
            self.eventOffset -= delta
        self.updateViewers()

    def seekEvent(self):
        delta = self.eventSeek.value()/1000000-(self.t+self.eventOffset)
        if self.isVideoPlaying.isChecked():
            self.t += delta
        else:
            self.eventOffset += delta
        self.updateViewers()

    def updateViewers(self):
        self.videoViewer.update(self.t)
        self.eventViewer.update(self.t+self.eventOffset)
        self.info.setText('v:{:.4f},d:{:.4f},e:{:.4f}'.format(self.t, self.eventOffset, self.t+self.eventOffset))

        videoBlocker = QSignalBlocker(self.videoSeek)
        eventBlocker = QSignalBlocker(self.eventSeek)
        self.videoSeek.setSliderPosition(int(self.t*1000000))
        self.eventSeek.setSliderPosition(int((self.t+self.eventOffset)*1000000))

    def positionChanged(self):
        if self.isVideoPlaying.isChecked():
            self.t += self.dt
        else:
            self.eventOffset += self.dt
        self.updateViewers()

    def play(self):
        if self.isPlaying:
            self.isPlaying = False
            self.timer.stop()
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        else:
            self.isPlaying = True
            self.timer.start(1000//60)
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

    def prev(self):
        print(self.t)
        if self.isVideoPlaying.isChecked():
            self.t -= self.dt
        else:
            self.eventOffset -= self.dt
        self.updateViewers()

    def next(self):
        print(self.t)
        if self.isVideoPlaying.isChecked():
            self.t += self.dt
        else:
            self.eventOffset += self.dt
        self.updateViewers()

    def seekStart(self):
        print(self.t)
        self.t = 0
        self.updateViewers()

    def seekEnd(self):
        print(self.t)
        self.t = (self.videoViewer.NFRAMES-1)//VIDEO_FPS
        self.updateViewers()

videoplayer = VideoPlayer()
videoplayer.resize(1200, 480)
videoplayer.show()

sys.exit(app.exec_())
