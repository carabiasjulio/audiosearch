from utils import *

import wx
import soundfile as sf
import sounddevice2 as sd
import numpy as np

import model
from model import SearchModel


class SearchFrame(wx.Frame):
    def __init__(self, model):
        """ model: SearchModel object """
        self.model = model

        wx.Frame.__init__(self, None, title='Search Audio by Example', size=(950, 700))
        self.Center()
        
        # main sizer/panel
        self.toppanel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(sizer)

        # query panel
        qsizer = wx.BoxSizer()
        qbox = wx.StaticBoxSizer(wx.StaticBox(self, label= 'Query examples'))
        qpanel = QueryPanel(self, model)
        uploadButton = wx.Button(self, label='upload')
        uploadButton.Bind(wx.EVT_BUTTON, qpanel.OnUpload)
        qbox.Add(uploadButton, 1, flag=wx.EXPAND|wx.ALL|wx.ALIGN_CENTRE_VERTICAL, border=2)
        qbox.Add(qpanel,8, flag=wx.EXPAND|wx.ALL, border=2)
        qsizer.Add(qbox, 6, wx.EXPAND|wx.ALL, border=5)
        goButton = wx.Button(self, label='SEARCH')
        goButton.Bind(wx.EVT_BUTTON, self.OnGo)
        qsizer.Add(goButton,1, flag=wx.EXPAND|wx.ALIGN_CENTRE_VERTICAL|wx.ALL, border= 10)

        # model control
#        controlBox = wx.StaticBoxSizer(wx.StaticBox(self, label='model control'))
#        m1 = wx.Button(self, label='mean distance ratio')
#        m1.Bind(wx.EVT_BUTTON, self.OnControl1)
#        m2 = wx.Button(self, label='K Nearest Neighbors')
#        m2.Bind(wx.EVT_BUTTON, self.OnControl2)
#        m3 = wx.Button(self, label='Naive Bayes')
#        m3.Bind(wx.EVT_BUTTON, self.OnControl3)
#        controlBox.Add(m1)
#        controlBox.Add(m2)
#        controlBox.Add(m3)

        model_options = ['mean distance ratio', 'K Nearest Neighbor', 'Naive Bayes']
        modelControl = wx.RadioBox(self, label='Model options', choices=model_options)
        self.modelControl = modelControl

        # ranking (results) panel
        rpanel = RankPanel(self, model)
        rpanel.Bind(wx.EVT_BUTTON, self.OnFeedback)
        self.rpanel = rpanel
        rbox = wx.StaticBoxSizer(wx.StaticBox(self, label='Results'))
        rbox.Add(rpanel, 1,flag=wx.EXPAND|wx.ALIGN_CENTRE|wx.ALL, border=2)

        # feedback panels 
        ysizer = wx.BoxSizer(wx.VERTICAL)
        yheader = wx.BoxSizer()
        yeslabel = wx.StaticText(self, label='Accepted:')
        clearYesButton = wx.Button(self, label='clear')
        clearYesButton.Bind(wx.EVT_BUTTON, self.OnClearYes)
        yesPanel = FeedbackPanel(self, model, True)
        yheader.Add(yeslabel, flag=wx.ALL, border=5)
        yheader.AddSpacer(80)
        yheader.Add(clearYesButton,flag=wx.BOTTOM, border=5)
        ysizer.Add(yheader,1)
        ysizer.Add(yesPanel,9,wx.EXPAND)

        nsizer = wx.BoxSizer(wx.VERTICAL)
        nheader = wx.BoxSizer()
        nolabel = wx.StaticText(self, label='Rejected:')
        clearNoButton = wx.Button(self, label='clear')
        clearNoButton.Bind(wx.EVT_BUTTON, self.OnClearNo)
        noPanel = FeedbackPanel(self, model, False)
        nheader.Add(nolabel,flag=wx.ALL, border=5)
        nheader.AddSpacer(80)
        nheader.Add(clearNoButton,flag=wx.BOTTOM, border=5)
        nsizer.Add(nheader,1)
        nsizer.Add(noPanel,9,wx.EXPAND)

        self.yesPanel = yesPanel
        self.noPanel = noPanel

        fbox = wx.StaticBoxSizer(wx.StaticBox(self, label='Feedback'))
        fbox.Add(ysizer,1, wx.EXPAND|wx.ALL, border=5)
        fbox.Add(wx.StaticLine(self,style=wx.LI_VERTICAL), flag=wx.EXPAND)
        fbox.Add(nsizer, 1,wx.EXPAND| wx.ALL, border=5)

        # global layout
        lowerSizer = wx.BoxSizer()
        lowerSizer.AddSpacer(5)
        lowerSizer.Add(rbox, 6, wx.EXPAND|wx.ALL, border=10)
        lowerSizer.Add(fbox, 10, wx.EXPAND|wx.ALL, border=10)
        lowerSizer.AddSpacer(5)

        sizer.Add(qsizer, 2, wx.EXPAND|wx.ALL, border=10)
        sizer.Add(modelControl, 1, wx.EXPAND|wx.ALL, border=15)
        sizer.Add(lowerSizer, 7, wx.EXPAND)
        sizer.AddSpacer(10)

        self.toppanel.SetSizer(sizer)

    def OnGo(self, event):
        choice = model.SCORE_FUNCS[self.modelControl.GetSelection()]
        self.model.update_scores(score_func=choice)
        self.rpanel.showRanking()
    
    def OnFeedback(self, event):
        self.yesPanel.updateView()
        self.noPanel.updateView()
        event.GetEventObject().GetParent().Destroy()   #TODO: better handling

    def OnClearYes(self, event):
        self.yesPanel.sizer.DeleteWindows()
        self.model.remove_all_feedback(True)

    def OnClearNo(self, event):
        self.noPanel.sizer.DeleteWindows()
        self.model.remove_all_feedback(False)

    def OnControl1(self, event):
        self.model.score_func = model.mean_dist_ratio

    def OnControl2(self, event):
        self.model.score_func = model.p_knn

    def OnControl3(self, event):
        self.model.score_func = model.p_MNB

class QueryPanel(wx.ScrolledWindow):
    def __init__(self, parent, model):
        self.parent = parent
	wx.ScrolledWindow.__init__(self, parent, -1, style=wx.TAB_TRAVERSAL)
        self.model = model
	sizer = wx.BoxSizer()

        self.sizer = sizer
        self.SetSizer(sizer)
        self.SetScrollRate(1,1)
        self.SetBackgroundColour('white')


    def OnUpload(self, event):
        fpick = wx.FileDialog(self,'choose query example','','',wildcard='WAV files (*.wav)|*.wav|all files (*)|*',style= wx.FD_OPEN)
        if fpick.ShowModal() == wx.ID_OK:
            f = fpick.GetPath()
            self.sizer.Add(ExampleSampleItem(self, self.model, f), flag=wx.ALL|wx.ALIGN_CENTRE_VERTICAL, border=5)
            self.OnInnerSizeChanged()
            # update model
            self.model.add_example(f)

    def OnInnerSizeChanged(self):
        w, h= self.sizer.GetMinSize()
        self.SetVirtualSize((w,h))

class RankPanel(wx.ScrolledWindow):
    def __init__(self, parent, model):
        """ rankings: tuple (fnames, scores) where fnames is a numpy array of strings and scores is a numpy array of numeric values """
        self.model = model
        wx.ScrolledWindow.__init__(self, parent)
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer = sizer
        self.SetSizer(sizer)
        self.SetScrollRate(1,1)
        self.SetBackgroundColour('white')
        self.showRanking()

    def showRanking(self, batchsize = 5):
        """ batchsize: number of results to load each time/page """
        proposals = self.model.get_proposals(batchsize)
        self.sizer.DeleteWindows()
        for (f, f_ind, score) in proposals: 
            self.sizer.Add(ProposedSampleItem(self, self.model, f, f_ind, score), flag=wx.ALL, border=5)
        self.SetVirtualSize(self.sizer.GetMinSize())

class FeedbackPanel(wx.ScrolledWindow):
    def __init__(self, parent, model, class_label):
        ''' class_label: boolean indicating whether user accepts or rejects samples shown in the panel'''
        self.model = model
        self.class_label = class_label

        wx.ScrolledWindow.__init__(self, parent)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)


        self.ssizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.ssizer)

        self.SetScrollRate(1,1)
        self.SetBackgroundColour('white')

    def updateView(self):
        self.ssizer.DeleteWindows()
        for s, s_ind in self.model.get_feedback(self.class_label):
           self.ssizer.Add(FeedbackSampleItem(self, self.model, s, s_ind, self.class_label), flag=wx.ALL, border=5)
        self.SetVirtualSize(self.sizer.GetMinSize())


class SampleItem(wx.Panel):
    ''' A mini sample player that features play/stop control, plus self removal'''
    def __init__(self, parent, model, sampleFile):
        wx.Panel.__init__(self, parent, style=wx.BORDER_STATIC)
        sizer = wx.BoxSizer()
        self.model = model

        self.playButton = wx.BitmapButton(self, bitmap = wx.Bitmap('play_s.png'))   #TODO: shrink button size
        self.playButton.Bind(wx.EVT_BUTTON, self.OnPlay)
        
        label = wx.StaticText(self, label= os.path.split(sampleFile)[-1])
      
        sizer.Add(self.playButton, flag=wx.ALL|wx.ALIGN_CENTER_VERTICAL, border=5)
        sizer.Add(label, flag=wx.ALL|wx.ALIGN_CENTER_VERTICAL, border=5)
        self.SetSizerAndFit(sizer)
        self.sizer = sizer

        self.SetBackgroundColour('light blue')

        self.sampleFile = sampleFile
        self.playing = False 

    def OnPlay(self, event):
        if self.playing:
            #print "stopping"
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

        removeButton = wx.Button(self, label = 'X', size=(30,30))
        removeButton.Bind(wx.EVT_BUTTON, self.OnRemove)
        
        sizer.Add(removeButton, flag=wx.ALL, border=5)
        self.SetSizerAndFit(sizer)

    def OnRemove(self, event):
        p = self.GetParent()
        p.SetVirtualSize(p.sizer.GetMinSize())
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
    def __init__(self, parent, model, sampleFile, s_ind, score):
        SampleItem.__init__(self, parent, model, sampleFile)
        sizer = self.sizer
        self.sample_index = s_ind
        self.score = score

        yesButton = wx.BitmapButton(self, bitmap=wx.Bitmap('yes.png'))
        yesButton.Bind(wx.EVT_BUTTON, self.OnYes)
        noButton = wx.BitmapButton(self, bitmap=wx.Bitmap('no.png'))
        noButton.Bind(wx.EVT_BUTTON, self.OnNo)
        bsizer = wx.BoxSizer()
        bsizer.Add(yesButton, flag=wx.ALIGN_CENTER_VERTICAL|wx.TOP|wx.BOTTOM, border=5)
        bsizer.Add(noButton, flag=wx.ALIGN_CENTER_VERTICAL|wx.TOP|wx.BOTTOM, border=5)
#       bsizer.Add(wx.StaticText(self, label="%.3f"%self.score), flag=wx.Top|wx.BOTTOM, border=5)
        bsizer.AddSpacer(5)
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


def main():
    app = TestApp()
    app.MainLoop()
if __name__ == '__main__':
    main()
