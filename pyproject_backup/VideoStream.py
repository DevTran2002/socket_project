__author__ = 'Tibbers'
import struct
import cv2
import numpy as np
import sys
import traceback
class VideoStream:
        def __init__(self, filename):
                self.Capture=cv2.VideoCapture(filename)
                if(self.Capture.isOpened() == False):
                        print("Got an error when opens the File")
                        traceback.print_exc(file=sys.stdout)
                self.filename = filename
                try:
                        self.file = open(filename, 'rb')
                        print ('-'*60 +  "\nVideo file : |" + filename +  "| read\n" + '-'*60)
                except:
                        print ("read " + filename + " error")
                        traceback.print_exc(file=sys.stdout)
                        raise IOError
                self.frameNum=0
        def nextFrame(self):
                data = self.file.read(5) # Get the framelength from the first 5 bytes
		#data_ints = struct.unpack('<' + 'B'*len(data),data)
                data = bytearray(data)
                data_int = (data[0] - 48) * 10000 + (data[1] - 48) * 1000 + (data[2] - 48) * 100 + (data[3] - 48) * 10 + (data[4] - 48)# = #int(data.encode('hex'),16)
                final_data_int = data_int
                if data:
                        framelength = final_data_int#int(data)#final_data_int/8  # xx bytes
                        # Read the current frame
                        frame = self.file.read(framelength)
                        if len(frame) != framelength:
                                raise ValueError('incomplete frame data')
                        self.frameNum += 1
                        print('-'*10 + "\nNext Frame (#" + str(self.frameNum) + ") length:" + str(framelength) + "\n" + '-'*10)
                        return frame
        def NewnextFrame(self):
                check,frame = self.Capture.read()
                if(self.Capture.isOpened()):
                        if check == True:
                                self.frameNum += 1
                                img_encode = cv2.imencode('.jpg',frame)[1]
                                data_encode = np.array(img_encode)
                                byte_encode = bytearray(data_encode.tobytes())
                                print("Next frame #"+str(self.frameNum)+" Length: "+str(len(byte_encode)))
                                return (True,byte_encode)
                        else:
                                traceback.print_exc(file=sys.stdout)
                                self.Capture.release()
                                return (False,frame)
                else:
                        traceback.print_exc(file=sys.stdout)
                        print("Capture closed")
                        return (False,frame)
        def DecodeFrame(raw_frame):
                tempt = np.frombuffer(raw_frame,dtype = np.uint8)
                h = tempt[0]*256+tempt[1]
                w = tempt[2]*256+tempt[3]
                image = np.zeros((h,w,3))
                idex = 4
                for i in range(0,h):
                         for j in range(0,w):
                                 for d in range(0,3):
                                         image[i,j][d] = tempt[idex]
                                         idex += 1
                return image.astype(np.uint8)
        def frameNbr(self):
                return self.frameNum
