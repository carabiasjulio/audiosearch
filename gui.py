from utils import *

import wx
import soundfile as sf
import sounddevice2 as sd
import numpy as np

import model
from model import SearchModel


# command IDs for buttons
ID_accept = 1
ID_reject = 0
ID_remove = 2

class SearchFrame(wx.Frame):
    def __init__(self, model):
        """ model: SearchModel object """
        self.model = model
        
        

        wx.Frame.__init__(self, None, title='Search Audio by Example', size=(950, 700))
        self.Center()
        
        # main sizer/panel
#        self.toppanel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(sizer)

        # query panel
        qsizer = wx.BoxSizer()
        qbox = wx.StaticBoxSizer(wx.StaticBox(self, label= 'Query examples'))
        qpanel = QueryPanel(self, model)
        uploadButton = wx.Button(self, label='upload', size=(70,40))
        uploadButton.Bind(wx.EVT_BUTTON, qpanel.OnUpload)
        qbox.Add(uploadButton,flag=wx.TOP|wx.BOTTOM|wx.ALIGN_CENTRE_VERTICAL, border=23)
        qbox.Add(qpanel,1, flag=wx.EXPAND|wx.ALL, border=2)

        qsizer.Add(qbox,1, flag=wx.ALL, border=5)
        goButton = wx.Button(self, label='SEARCH', size=(80,-1))
        goButton.Bind(wx.EVT_BUTTON, self.OnGo)
        qsizer.Add(goButton,flag=wx.EXPAND|wx.ALIGN_CENTRE_VERTICAL|wx.ALL, border= 10)

        sizer.Add(qsizer, flag= wx.EXPAND|wx.ALL, border=10)

        ### model control
        kLabel = wx.StaticText(self, label='Number of neighbors:')
        # number of neighbors
        kChoice = wx.SpinCtrl(self, min=1, max=12, initial=1)
        self.kChoice = kChoice
        kChoice.Disable()

        csizer = wx.BoxSizer()
        csizer.AddSpacer(12)
        csizer.Add(kLabel, flag=wx.ALL|wx.ALIGN_CENTER, border=5)
        csizer.Add(kChoice, flag=wx.ALL|wx.ALIGN_CENTER, border=5)
        
        sizer.Add(csizer, flag=wx.EXPAND, border=5)
        #model_options = ['mean distance ratio', 'K Nearest Neighbor', 'Naive Bayes']
        #modelControl = wx.RadioBox(self, label='Model options', choices=model_options)
        #self.modelControl = modelControl
        #sizer.Add(modelControl, 1, wx.EXPAND|wx.ALL, border=15)
        
        ### ranking (results) panel

        rpanel = RankPanel(self, model)
        rpanel.Bind(wx.EVT_BUTTON, self.OnFeedback, id=ID_accept)
        rpanel.Bind(wx.EVT_BUTTON, self.OnFeedback, id=ID_reject)
        self.rpanel = rpanel
        rbox = wx.StaticBoxSizer(wx.StaticBox(self, label='Results'))
        rbox.Add(rpanel, 1,flag=wx.EXPAND|wx.ALIGN_CENTRE|wx.ALL, border=2)

        # feedback panels 
        ysizer = wx.BoxSizer(wx.VERTICAL)
        yheader = wx.BoxSizer()
        yeslabel = wx.StaticText(self, label='Accepted:')
        yesCount = wx.StaticText(self, label="(0)")
        yesCount.SetForegroundColour('red')
        clearYesButton = wx.Button(self, id = ID_accept, label='clear', size=(60,30))
        clearYesButton.Bind(wx.EVT_BUTTON, self.OnClearFeedback)
        yesPanel = FeedbackPanel(self, model, True)
        yesPanel.Bind(wx.EVT_BUTTON, self.OnRemoveFeedback, id=ID_remove)
        yheader.Add(yeslabel, flag=wx.ALL, border=5)
        yheader.Add(yesCount, flag=wx.ALL, border=5)
        yheader.AddStretchSpacer()
        yheader.Add(clearYesButton,flag=wx.BOTTOM, border=5)
        ysizer.Add(yheader,1, wx.EXPAND)
        ysizer.Add(yesPanel,9,wx.EXPAND)

        nsizer = wx.BoxSizer(wx.VERTICAL)
        nheader = wx.BoxSizer()
        nolabel = wx.StaticText(self, label='Rejected:')
        noCount = wx.StaticText(self, label="(0)")
        clearNoButton = wx.Button(self, id=ID_reject, label='clear', size=(60,30))
        clearNoButton.Bind(wx.EVT_BUTTON, self.OnClearFeedback)
        noPanel = FeedbackPanel(self, model, False)
        noPanel.Bind(wx.EVT_BUTTON, self.OnRemoveFeedback, id=ID_remove)
        nheader.Add(nolabel,flag=wx.ALL, border=5)
        nheader.Add(noCount, flag=wx.ALL, border=5)
        nheader.AddStretchSpacer()
        nheader.Add(clearNoButton,flag=wx.BOTTOM, border=5)
        nsizer.Add(nheader,1, wx.EXPAND)
        nsizer.Add(noPanel,9,wx.EXPAND)

        self.feedbackCounts = (noCount, yesCount)
        self.feedbackPanels = (noPanel, yesPanel)

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

        sizer.Add(lowerSizer, 7, wx.EXPAND)
        sizer.AddSpacer(10)


    def OnNewTask(self, event):
        # Check if current task is completed TODO
        if not self.model.task_completed():
            dlg = wx.MessageDialog(self, 'The current task is not completed yet. Are you sure you want to quit?', 'Current task not completed')
            r = dlg.ShowModal()
            dlg.Destroy()
            if r!=wx.ID_OK:
                return

        # prompt user to choose target class
        dialog = wx.SingleChoiceDialog(self, "Choose a target sound to search","Choose search target", choices = CLASS_NAMES)
        dialog.ShowModal()
        choice = dialog.GetSelection()
        dialog.Destroy()

        # update model
        self.model.restart()
        self.model.set_target_class(choice)
        
        # update taskLabel
        self.taskLabel.SetLabel("Target: %s" % CLASS_NAMES[choice])

        # retrieve target example
        s_ind, sampleFile = self.model.get_target_example()
        self.examplePane.DestroyChildren()
        sizer = self.examplePane.GetSizer()
        sizer.Add(SampleItem(self.examplePane, self.model, sampleFile), 0,flag=wx.ALIGN_CENTER|wx.ALL, border=5)
        sizer.Layout()
        
        # clear previous results and feedback
        self.rpanel.DestroyChildren()
        [p.DestroyChildren() for p in self.feedbackPanels]

    def OnGo(self, event):
        k = self.kChoice.GetValue()
        self.rpanel.DestroyChildren()
        self.model.update_scores(k)
        self.rpanel.showRanking()
   
   # Feedback event handlers
    def OnFeedback(self, event):
        c = event.GetEventObject().GetId()
        self.feedbackPanels[c].updateView()
        event.GetEventObject().GetParent().Destroy()   #TODO: better handling
        self.rpanel.refresh()
        self.updateFeedbackCount(c)
        self.updateKChoice()

        if c==0:
            self.kChoice.Enable()

    def OnRemoveFeedback(self, event):
        # get sample item
        s = event.GetEventObject().GetParent()
        # get feedback panel
        p = s.GetParent()
        # update model
        c = s.class_label
        self.model.remove_feedback(c, s.sample_index)
        # update feedback count label
        self.updateFeedbackCount(c)
        # remove from results
        s.Destroy()
        p.refresh()
        self.updateKChoice()

        if c==0:
            if len(self.model.get_feedback(0))==0:
                self.kChoice.Disable()
        
    def OnClearFeedback(self, event):
        c = event.GetId()
        self.feedbackPanels[c].sizer.DeleteWindows()
        self.model.remove_all_feedback(c)
        self.updateFeedbackCount(c)
        self.updateKChoice()
        if c==0:
            if len(self.model.get_feedback(0))==0:
                self.kChoice.Disable()

    def updateFeedbackCount(self, class_label):
        countLabel = self.feedbackCounts[class_label]
        countLabel.SetLabel("(%d)" % len(self.model.get_feedback(class_label)))
        if class_label:
            if self.model.task_completed():
                countLabel.SetForegroundColour('green')
            else:
                countLabel.SetForegroundColour('red')

    def updateKChoice(self):
        n = self.model.get_trainset_size()
        self.kChoice.SetValue(min(3,n))
        self.kChoice.SetRange(1, n)

    def OnControl1(self, event):
        self.model.score_func = model.mean_dist_ratio

    def OnControl2(self, event):
        self.model.score_func = model.p_knn

    def OnControl3(self, event):
        self.model.score_func = model.p_MNB

class QueryPanel(wx.ScrolledWindow):
    def __init__(self, parent, model):
        self.model = model
	wx.ScrolledWindow.__init__(self, parent, -1, style=wx.TAB_TRAVERSAL, size=(200, 400))
        
	sizer = wx.BoxSizer()
        self.sizer = sizer
        self.SetSizer(sizer)

        self.SetScrollRate(1,1)
        self.SetScrollPageSize(wx.HORIZONTAL,150)
        self.SetBackgroundColour('white')

        self.Bind(wx.EVT_BUTTON, self.OnRemove, id=ID_remove)     # only remove event is skipped

    def OnUpload(self, event):
        fpick = wx.FileDialog(self,'choose query example','','',wildcard='WAV files (*.wav)|*.wav|all files (*)|*',style= wx.FD_OPEN)
        if fpick.ShowModal() == wx.ID_OK:
            f = fpick.GetPath()
            e = ExampleSampleItem(self, self.model, f)
            self.sizer.Add(e, flag=wx.LEFT|wx.ALIGN_CENTRE_VERTICAL, border=5)
            self.OnInnerSizeChanged()
            # update model
            self.model.add_example(f)
        fpick.Destroy()

    def OnInnerSizeChanged(self):
        w, h= self.sizer.GetMinSize()
        self.SetVirtualSize((w,h))

    def OnRemove(self, event):
        s = event.GetEventObject().GetParent()
        self.model.remove_example(s.sampleFile)
        s.Destroy()
        self.Layout()

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
#        self.showRanking()

    def showRanking(self, batchsize = 5):
        """ batchsize: number of results to load each time/page """
        proposals = self.model.get_proposals(batchsize)
        #self.sizer.DeleteWindows()
        for (f, f_ind, score) in proposals: 
            self.sizer.Add(ProposedSampleItem(self, self.model, f, f_ind, score), flag=wx.TOP, border=5)
        self.SetVirtualSize(self.sizer.GetMinSize())
    
    def refresh(self):
        self.Layout()
        self.SetVirtualSize(self.GetMinSize()) 

class FeedbackPanel(wx.ScrolledWindow):
    def __init__(self, parent, model, class_label):
        ''' class_label: boolean indicating whether user accepts or rejects samples shown in the panel'''
        self.model = model
        self.class_label = class_label

        wx.ScrolledWindow.__init__(self, parent, -1, style=wx.TAB_TRAVERSAL)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)

        self.SetScrollRate(1,1)
        self.SetBackgroundColour('white')
      
    def updateView(self):
        print 'Update feedback'
        self.sizer.DeleteWindows()
        for s, s_ind in self.model.get_feedback(self.class_label):
            sample = FeedbackSampleItem(self, self.model, s, s_ind, self.class_label)
            self.sizer.Add(sample, flag=wx.TOP, border=5)
        self.SetVirtualSize(self.sizer.GetMinSize())

    def refresh(self):
        self.Layout()
        self.SetVirtualSize(self.GetMinSize())


class SampleItem(wx.Panel):
    ''' A mini sample player that features play/stop control, plus self removal'''
    def __init__(self, parent, model, sampleFile):
        wx.Panel.__init__(self, parent, style=wx.BORDER_STATIC)
        sizer = wx.BoxSizer()
        self.model = model

        self.playButton = wx.BitmapButton(self, bitmap = wx.Bitmap('img/play_s.png'))   #TODO: shrink button size
        self.playButton.Bind(wx.EVT_BUTTON, self.OnPlay)
        
        label = wx.StaticText(self, label= os.path.split(sampleFile)[-1], size=(140,-1))
      
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
            self.playButton.SetBitmapLabel(bitmap = wx.Bitmap('img/stop_s.png'))
            self.playing = True

    def AfterPlay(self):
        self.playButton.SetBitmapLabel(bitmap = wx.Bitmap('img/play_s.png'))
        self.playing = False        

       
class RemovableSampleItem(SampleItem):
    """ SampleItem that can be removed """
    def __init__(self, parent, model, sampleFile):
        SampleItem.__init__(self, parent, model, sampleFile)
        sizer = self.sizer

        removeButton = wx.Button(self, id = ID_remove, label = 'X', size=(30,30))
        removeButton.Bind(wx.EVT_BUTTON, self.OnRemove)
        
        sizer.Add(removeButton, flag=wx.ALL, border=5)
        self.SetSizerAndFit(sizer)

    def OnRemove(self, event):
#        p = self.GetParent()
#        p.SetVirtualSize(p.sizer.GetMinSize())
#        self.updateModel()
#        p.updateView()
#        self.Destroy()
        event.Skip()


class ExampleSampleItem(RemovableSampleItem):
    pass

class FeedbackSampleItem(RemovableSampleItem):
    def __init__(self, parent, model, sampleFile, sample_index, class_label):
        RemovableSampleItem.__init__(self, parent, model, sampleFile)
        self.sample_index = sample_index
        self.class_label = class_label

class ProposedSampleItem(SampleItem):
    def __init__(self, parent, model, sampleFile, s_ind, score):
        SampleItem.__init__(self, parent, model, sampleFile)
        sizer = self.sizer
        self.sample_index = s_ind
        self.score = score

        yesButton = wx.BitmapButton(self, id=ID_accept, bitmap=wx.Bitmap('img/yes.png'))
        yesButton.Bind(wx.EVT_BUTTON, self.OnYes)
        noButton = wx.BitmapButton(self, id=ID_reject, bitmap=wx.Bitmap('img/no.png'))
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
    def __init__(self, user):
        self.user = user
        wx.App.__init__(self)

    def OnInit(self):
        w = SearchFrame(SearchModel(self.user))
        w.Show()
        self.SetTopWindow(w)
        return True

import sys

def main(user):
    app = TestApp(user)
    app.MainLoop()

if __name__ == '__main__':
    main(sys.argv[1])   # take the first argument as user name, ignore trailing stuff
