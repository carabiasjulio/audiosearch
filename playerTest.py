from utils import *
import wx
from wx import media as media

import soundfile as sf
import sounddevice2 as sd
reload(sd)

class SearchFrame(wx.Frame):
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
    

class QueryPanel(wx.Panel):
	def __init__(self, parent):
		wx.Panel.__init__(self, parent)
		
		sizer = wx.Grid

        
class SampleItem(wx.Panel):
    ''' A mini sample player that features play/stop control, plus self removal'''
    def __init__(self, sampleFile, parent):
        wx.Panel.__init__(self, parent)
        
        sizer = wx.BoxSizer()

        self.playButton = wx.BitmapButton(self, bitmap = wx.Bitmap('play.png'))   #TODO: shrink button size
        self.playButton.Bind(wx.EVT_BUTTON, self.OnPlay)
        
        label = wx.StaticText(self, label= os.path.split(sampleFile)[-1])
      
        
        #self.removeButton = wx.BitmapButton(self, bitmap = wx.Bitmap('remove.png'))
        self.removeButton = wx.Button(self, label = 'X')
        self.removeButton.Bind(wx.EVT_BUTTON, self.OnRemove)
        
        sizer.Add(self.playButton)
        sizer.Add(label)
        sizer.Add(self.removeButton)
        self.SetSizerAndFit(sizer)
        self.sizer = sizer

        self.sampleFile = sampleFile
        self.playing = False 
        
    def OnRemove(self, event):
        self.Destroy()

    def OnMediaStop(self, event):
        print 'stopped!'
        btn = self.playButton
        btn.SetBitmapLabel(bitmap = wx.Bitmap('play.png'))
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
            #sd.wait()
            #self.AfterPlay()

    def AfterPlay(self):
        self.playButton.SetBitmapLabel(bitmap = wx.Bitmap('play.png'))
        self.playing = False        

        
        
class FeedbackSampleItem(SampleItem):
    """ Sample bar with label control"""
    def __init__(self, sampleFile, parent):
        SampleItem.__init__(self, sampleFile, parent)
        rbsizer = wx.BoxSizer()
        rb1 = wx.RadioButton(self, label='yes', style=wx.RB_GROUP)
        rb2 = wx.RadioButton(self, label='no')
        rb3 = wx.RadioButton(self, label='unknown')
        rbsizer.Add(rb1)
        rbsizer.Add(rb2)
        rbsizer.Add(rb3)
        self.sizer.Add(rbsizer, flag=wx.EXPAND)
        self.SetSizerAndFit(self.sizer)
        
        rb1.Bind(wx.EVT_RADIOBUTTON, self.OnYes)
        rb2.Bind(wx.EVT_RADIOBUTTON, self.OnNo)
        rb3.Bind(wx.EVT_RADIOBUTTON, self.OnUnknown)
        
    def OnYes(self, event):
        # TODO: update model
        pass
    def OnNo(self, event):
        # TODO: update model
        pass
    def OnUnknown(self, event):
        # TODO: update model
        pass
       

class TestApp(wx.App):    
    def OnInit(self):
        w = SearchFrame()
        w.Show()
        self.SetTopWindow(w)
        return True
        
app = TestApp()
app.MainLoop()
