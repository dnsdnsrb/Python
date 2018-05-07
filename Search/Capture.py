import os
import sys
import time
import image
import ImageGrab
#---------------------------------------------------------
#User Settings:
SaveDirectory=r'C:\Documents and Settings\Gregory\Desktop'
ImageEditorPath=r'C:\WINDOWS\system32\mspaint.exe'
#Here is another example:
#ImageEditorPath=r'C:\Program Files\IrfanView\i_view32.exe'
#---------------------------------------------------------

img=ImageGrab.grab()
saveas=os.path.join(SaveDirectory,'ScreenShot_'+time.strftime('%Y_%m_%d%_%H_%M_%S')+'.jpg')
img.save(saveas)
editorstring='""%s" "%s"'% (ImageEditorPath,saveas) #Just for Windows right now?
#Notice the first leading " above? This is the bug in python that no one will admit...
os.system(editorstring)
#ㅇㄹㄴㅇㄹㄴ