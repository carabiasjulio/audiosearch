from utils import *
import wx
from wx import media as media

import soundfile as sf
import sounddevice2 as sd
import numpy as np


class SampleFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, title='Test')
        
        
        samplePanel = wx.ScrolledWindow(self, style=wx.TAB_TRAVERSAL)       
        psizer = wx.BoxSizer(wx.VERTICAL) 
        # clamped input       
        psizer.Add(FeedbackSampleItem(random_audio_file(), samplePanel))
        psizer.Add(FeedbackSampleItem(random_audio_file(), samplePanel))
        psizer.Add(FeedbackSampleItem(random_audio_file(), samplePanel))
        
        samplePanel.SetSizer(psizer)
        samplePanel.EnableScrolling(True, True)
        samplePanel.SetScrollbars(1,1,1,1)
        
        sizer = wx.BoxSizer()
        sizer.Add(samplePanel, 1, wx.EXPAND)
        self.SetSizer(sizer)
    

class SearchFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, title='Search Audio by Example', size=(800,800))
        self.Center()
        sizer = wx.BoxSizer(wx.VERTICAL)
        qpanel = QueryPanel(self)
        nfiles = 100
        r = (np.array([random_audio_file() for i in range(nfiles)]), np.random.rand(nfiles))
        rpanel = RankPanel(self, r)
        sizer.Add(qpanel)
        sizer.Add(rpanel)
        self.SetSizerAndFit(sizer)

class QueryPanel(wx.Panel):
    def __init__(self, parent):
	wx.Panel.__init__(self, parent)
	sizer = wx.BoxSizer()
        uploadButton = wx.Button(label='upload', parent=self)
        sizer.Add(uploadButton)
        self.SetSizer(sizer)
        self.sizer = sizer
        uploadButton.Bind(wx.EVT_BUTTON, self.OnUpload)

    def OnUpload(self, event):
        fpick = wx.FileDialog(self,'choose query example','','',wildcard='WAV files (*.wav)|*.wav|all files (*)|*',style= wx.FD_OPEN)
        if fpick.ShowModal() == wx.ID_OK:
            f = fpick.GetPath()
            self.sizer.Add(RemovableSampleItem(f, self))
            self.Fit()
            self.GetParent().Fit()
        
class RankPanel(wx.Panel):
    def __init__(self, parent, ranking):
        """ rankings: tuple (fnames, scores) where fnames is a numpy array of strings and scores is a numpy array of numeric values """
        wx.Panel.__init__(self, parent)
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer = sizer
        self.SetSizer(sizer)
        self.ranking = ranking
        self.showRanking()

    def updateRanking(self, newRanking):
        self.ranking = newRanking
        self.showRanking()

    def showRanking(self, batchsize = 5):
        """ batchsize: number of results to load each time/page """
        fnames, scores = self.ranking
        sorting = np.argsort(scores)
        print len(fnames[sorting])
        fshow = fnames[sorting][:batchsize]
        for f in fshow:
            self.sizer.Add(ProposedSampleItem(f, self))
        self.Fit()

class FeedbackPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

class SampleItem(wx.Panel):
    ''' A mini sample player that features play/stop control, plus self removal'''
    def __init__(self, sampleFile, parent):
        wx.Panel.__init__(self, parent)
        sizer = wx.BoxSizer()

        self.playButton = wx.BitmapButton(self, bitmap = wx.Bitmap('play.png'))   #TODO: shrink button size
        self.playButton.Bind(wx.EVT_BUTTON, self.OnPlay)
        
        label = wx.StaticText(self, label= os.path.split(sampleFile)[-1])
      
        sizer.Add(self.playButton)
        sizer.Add(label)
        self.SetSizerAndFit(sizer)
        self.sizer = sizer

        self.sampleFile = sampleFile
        self.playing = False 

    def OnPlay(self, event):
        if self.playing:
            print "stopping"
            sd.stop()
            self.AfterPlay()
            
        else:
            f = self.sampleFile
            print f
            data, fs = sf.read(f)
            sd.play(data, fs, do_after = self.AfterPlay)    # new thread, with 
            self.playButton.SetBitmapLabel(bitmap = wx.Bitmap('stop.png'))
            self.playing = True

    def AfterPlay(self):
        self.playButton.SetBitmapLabel(bitmap = wx.Bitmap('play.png'))
        self.playing = False        

       
class RemovableSampleItem(SampleItem):
    """ SampleItem that can be removed """
    def __init__(self, sampleFile, parent):
        SampleItem.__init__(self, sampleFile, parent)
        sizer = self.sizer

        removeButton = wx.Button(self, label = 'X')
        removeButton.Bind(wx.EVT_BUTTON, self.OnRemove)
        
        sizer.Add(removeButton)
        self.Fit()

    def OnRemove(self, event):
        p = self.GetParent()
        self.Destroy()
        p.Fit()     # update panel size

class ProposedSampleItem(SampleItem):
    def __init__(self, sampleFile, parent):
        SampleItem.__init__(self, sampleFile, parent)
        sizer = self.sizer

        yesButton = wx.BitmapButton(self, bitmap=wx.Bitmap('yes.png'))
        yesButton.Bind(wx.EVT_BUTTON, self.OnYes)
        noButton = wx.BitmapButton(self, bitmap=wx.Bitmap('no.png'))
        noButton.Bind(wx.EVT_BUTTON, self.OnNo)
        bsizer = wx.BoxSizer()
        bsizer.Add(yesButton)
        bsizer.Add(noButton)
        self.sizer.Add(bsizer)
        self.SetSizerAndFit(self.sizer)

    def OnYes(self, event):
        print "accepted:", self.sampleFile
        # TODO: update feedback panel
        self.Destroy()

    def OnNo(self, event):
        print "rejected:", self.sampleFile
        #TODO: update feedback
        self.Destroy()
        
class TestApp(wx.App):    
    def OnInit(self):
        w = SearchFrame()
        w.Show()
        self.SetTopWindow(w)
        return True
        
app = TestApp()
app.MainLoop()
