from utils import *

import wx
import soundfile as sf
import sounddevice2 as sd
import numpy as np

from model import SearchModel

class SearchFrame(wx.Frame):
    def __init__(self, model):
        """ model: SearchModel object """
        self.model = model

        wx.Frame.__init__(self, None, title='Search Audio by Example', size=(800,800))
        self.Center()
        
        # query panel
        qpanel = QueryPanel(self, model)
        
        goButton = wx.Button(qpanel, label='GO')
        qpanel.Bind(wx.EVT_BUTTON, self.OnGo)
        qpanel.sizer.Add(goButton)


        # ranking (results) panel
#        nfiles = 100
#        r = (np.array([random_audio_file() for i in range(nfiles)]), np.random.rand(nfiles))
        rpanel = RankPanel(self, model)
        rpanel.Bind(wx.EVT_BUTTON, self.OnFeedback)
        self.rpanel = rpanel

        # feedback panels 

        yesPanel = FeedbackPanel(self, model, True)
        noPanel = FeedbackPanel(self, model, False)
        self.yesPanel = yesPanel
        self.noPanel = noPanel

        # layout
        lowerSizer = wx.BoxSizer()
        lowerSizer.Add(yesPanel)
        lowerSizer.Add(rpanel)
        lowerSizer.Add(noPanel)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(qpanel)
        sizer.Add(lowerSizer)

        self.SetSizerAndFit(sizer)

    def OnGo(self, event):
        self.model.update_scores()
        self.rpanel.showRanking()
        self.Fit()
    
    def OnFeedback(self, event):
        self.yesPanel.updateView()
        self.noPanel.updateView()
        event.GetEventObject().GetParent().Destroy()   #TODO: better handling
        self.Fit()


class QueryPanel(wx.Panel):
    def __init__(self, parent, model):
	wx.Panel.__init__(self, parent)
        self.model = model
	sizer = wx.BoxSizer()
        uploadButton = wx.Button(label='upload', parent=self)
        uploadButton.Bind(wx.EVT_BUTTON, self.OnUpload)
        sizer.Add(uploadButton)

        self.sizer = sizer
        self.SetSizer(sizer)

    def OnUpload(self, event):
        fpick = wx.FileDialog(self,'choose query example','','',wildcard='WAV files (*.wav)|*.wav|all files (*)|*',style= wx.FD_OPEN)
        if fpick.ShowModal() == wx.ID_OK:
            f = fpick.GetPath()
            self.sizer.Add(ExampleSampleItem(self, self.model, f))
            self.Fit()
            self.GetParent().Fit()
            # update model
            self.model.add_example(f)

class RankPanel(wx.Panel):
    def __init__(self, parent, model):
        """ rankings: tuple (fnames, scores) where fnames is a numpy array of strings and scores is a numpy array of numeric values """
        self.model = model
        wx.Panel.__init__(self, parent)
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer = sizer
        self.SetSizer(sizer)
        self.showRanking()

    def showRanking(self, batchsize = 5):
        """ batchsize: number of results to load each time/page """
        scores = self.model.scores
        sorting = np.argsort(scores)[::-1]  # descending order
        print 'proposing:', sorting[:batchsize]
        self.sizer.DeleteWindows()
        for f_ind in sorting[:batchsize]:
            f = get_libsample(f_ind)
            self.sizer.Add(ProposedSampleItem(self, self.model, f, f_ind))
        self.sizer.Layout()
        self.Fit()

class FeedbackPanel(wx.Panel):
    def __init__(self, parent, model, class_label):
        ''' class_label: boolean indicating whether user accepts or rejects samples shown in the panel'''
        self.model = model
        self.class_label = class_label
        wx.Panel.__init__(self, parent)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)
        if class_label:
            label = wx.StaticText(self, label='Yes!')
        else:
            label = wx.StaticText(self, label='NOOO')
        self.sizer.Add(label)

        clearButton = wx.Button(self, label='clear')
        clearButton.Bind(wx.EVT_BUTTON, self.OnClear)
        self.sizer.Add(clearButton)

        self.ssizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.ssizer)

    def updateView(self):
        print 'updating feedback list'
        self.ssizer.DeleteWindows()
        for s_ind in self.model.feedback[self.class_label]:
           self.ssizer.Add(FeedbackSampleItem(self, self.model, get_libsample(s_ind), s_ind, self.class_label))
        self.sizer.Layout()
        self.Fit()

    def OnClear(self, event):
        self.ssizer.DeleteWindows()
        self.model.remove_all_feedback(self.class_label)

class SampleItem(wx.Panel):
    ''' A mini sample player that features play/stop control, plus self removal'''
    def __init__(self, parent, model, sampleFile):
        wx.Panel.__init__(self, parent)
        sizer = wx.BoxSizer()
        self.model = model

        self.playButton = wx.BitmapButton(self, bitmap = wx.Bitmap('play_s.png'))   #TODO: shrink button size
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
            self.playButton.SetBitmapLabel(bitmap = wx.Bitmap('stop_s.png'))
            self.playing = True

    def AfterPlay(self):
        self.playButton.SetBitmapLabel(bitmap = wx.Bitmap('play_s.png'))
        self.playing = False        

       
class RemovableSampleItem(SampleItem):
    """ SampleItem that can be removed """
    def __init__(self, parent, model, sampleFile):
        SampleItem.__init__(self, parent, model, sampleFile)
        sizer = self.sizer

        removeButton = wx.Button(self, label = 'X')
        removeButton.Bind(wx.EVT_BUTTON, self.OnRemove)
        
        sizer.Add(removeButton)
        self.SetSizerAndFit(sizer)
#        self.Fit()

    def OnRemove(self, event):
        p = self.GetParent()
        p.Fit()     # update panel size
        self.updateModel()
        self.Destroy()

    def updateModel(self):
        pass


class ExampleSampleItem(RemovableSampleItem):
    def updateModel(self):
        self.model.remove_example(self.sampleFile)

class FeedbackSampleItem(RemovableSampleItem):
    def __init__(self, parent, model, sampleFile, sample_index, class_label):
        RemovableSampleItem.__init__(self, parent, model, sampleFile)
        self.sample_index = sample_index
        self.class_label = class_label

    def updateModel(self):
        self.model.remove_feedback(self.class_label, self.sample_index)

class ProposedSampleItem(SampleItem):
    def __init__(self, parent, model, sampleFile, s_ind):
        SampleItem.__init__(self, parent, model, sampleFile)
        sizer = self.sizer
        self.sample_index = s_ind

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
        self.model.add_feedback(True, self.sample_index)
        event.Skip()

    def OnNo(self, event):
        print "rejected:", self.sampleFile
        self.model.add_feedback(False, self.sample_index)
        event.Skip()
        
class TestApp(wx.App):    
    def OnInit(self):
        w = SearchFrame(SearchModel())
        w.Show()
        self.SetTopWindow(w)
        return True
        
app = TestApp()
app.MainLoop()
