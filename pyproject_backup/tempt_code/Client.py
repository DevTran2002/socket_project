from tkinter import *
from tkinter import messagebox as MessageBox
from PIL import Image, ImageTk
import socket, threading, sys, traceback,os
import cv2,numpy as np
from RtpPacket import RtpPacket
CACHE_FILE_NAME = "cache-"
CACHE_FILE_EXT =  ".jpg"
class Client:
    INIT = 0
    READY = 1
    PLAYING = 2
    state = INIT
    SETUP = 0
    PLAY = 1
    PAUSE = 2
    QUIT = 3
    counter = 0
    def __init__(self,control,serveraddr,serverport,rtpport,filename):
        self.control = control
        self.control.protocol("WM_DELETE_WINDOW",self.handler) #xu ly khi nguoi dung quit bang window manager
        self.createWidgets()
        self.serveraddr = serveraddr
        self.serverport = int(serverport)
        self.rtpport = int(rtpport)
        self.filename = filename
        self.rtspSeq = 0
        self.sessionId = 0
        self.requestSent = -1
        self.quitAcked = 0
        self.connectToServer()
        self.frameNumber= 0
        self.rtpSocket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    def createWidgets(self):
        self.setup = Button(self.control,width=20,padx=3,pady=3,text="Make session",command = self.setupVideo)
        self.setup.grid(row = 1,column = 0,padx = 2, pady = 2)
        
        self.start = Button(self.control,width=20,padx=3,pady=3,text="Start",command = self.playVideo)
        self.start.grid(row=1,column = 1,padx=2,pady=2)
        
        self.pause = Button(self.control,width=20,padx=3,pady=3,text="Pause",command = self.pauseVideo)
        self.pause.grid(row=1,column=2,padx=2,pady=2)
        
        self.quit = Button(self.control,width=20,padx=3,pady=3,text="Quit",command = self.quit)
        self.quit.grid(row=1,column=3,padx=2,pady=2)
        
        self.label = Label(self.control,height=19)
        self.label.grid(row=0,column=0,columnspan=4,sticky=W+E+N+S,padx=5,pady=5)
        
    def setupVideo(self):
        if self.state ==  self.INIT:
            print("test")
            self.sendRtspRequest(self.SETUP)
    def playVideo(self):
        if self.state == self.READY:
            print("Playing Video")
            threading.Thread(target = self.listenRtp).start()
            self.playEvent = threading.Event()
            self.playEvent.clear()
            self.sendRtspRequest(self.PLAY)
    def pauseVideo(self):
        print("hadle pause")
        if self.state == self.PLAYING:
            print("hadle pause")
            self.sendRtspRequest(self.PAUSE)
    def quit(self):
        self.sendRtspRequest(self.QUIT)
        self.control.destroy()
        #os.remove(CACHE_FILE_NAME+str(self.sessionId)+CACHE_FILE_EXT)
        rate = float(self.counter/self.frameNumber)
        print("Packet loss rate: "+str(rate))
        sys.exit(0)
    def listenRtp(self):
        while True:
            if cv2.waitKey(1) == ord('q'):
                break
            try:
                data,addr = self.rtpSocket.recvfrom(20480)
                if data:
                    rtpPacket = RtpPacket()
                    rtpPacket.decode(data)
                    currFrameNbr = rtpPacket.seqNum()
                    subseq = rtpPacket.subseq()
                    endmark =  rtpPacket.endmark()
                    if currFrameNbr > self.frameNumber and subseq == 0:
                        print("Display  video")
                        self.frameNumber = currFrameNbr
                        self.display(rtpPacket.getPayload())
                        #self.updateVideo(self.writeFrame(rtpPacket.getPayload()))
                    if subseq != 0:
                        if currFrameNbr > self.frameNumber:
                            if subseq == 1:
                                self.checksubseq = 1
                                self.frameNumber = currFrameNbr
                                self.buffer = bytearray(rtpPacket.getPayload())
                            else:
                                print("Got subPacket loss")
                        elif currFrameNbr == self.frameNumber and endmark ==0:
                            if subseq == self.checksubseq+1:
                                self.checksubseq+=1
                                self.buffer+=bytearray(rtpPacket.getPayload())
                            else:
                                print("Got subPacket loss")
                        elif currFrameNbr == self.frameNumber and endmark ==1:
                            if subseq == self.checksubseq+1:
                                self.checksubseq+=1
                                self.buffer+=bytearray(rtpPacket.getPayload())
                                print("Display  video")
                                self.display(self.buffer)
                                #self.updateVideo(self.writeFrame(rtpPacket.getPayload()))
                            else:
                                print("Got subPacket loss")
            except:
                print("Did not receive udp data")
                if self.playEvent.isSet():
                    cv2.destroyAllWindows()
                    break
                if self.quitAcked == 1:
                    self.rtpSocket.shutdown(socket.SHUT_RDWR)
                    self.rtpSocket.close()
                    break
        '''while True:
            if cv2.waitKey(1) == ord('q'):
                break
            try:
                data,addr = self.rtpSocket.recvfrom(20480)
                if data:
                    rtpPacket = RtpPacket()
                    rtpPacket.decode(data)
                    print("Received Packet Number: "+str(rtpPacket.seqNum()))
                    try:
                        if (self.frameNumber+1) != rtpPacket.seqNum():
                            self.counter += 1
                            print("Got Packet loss")
                        currFrameNbr = rtpPacket.seqNum()
                    except:
                        print("seqNum() error")
                        traceback.print_exc(file = sys.stdout)
                        
                    if currFrameNbr > self.frameNumber:
                        print("Display  video")
                        self.frameNumber = currFrameNbr
                        #self.updateVideo(self.writeFrame(rtpPacket.getPayload()))
                        self.display(rtpPacket.getPayload())
            except:
                #print("Did not receive udp data")
                if self.playEvent.isSet():
                    break
                if self.quitAcked == 1:
                    self.rtpSocket.shutdown(socket.SHUT_RDWR)
                    self.rtpSocket.close()
                    break'''
    def display(self,byte_data):
        print("test of display:")
        print("Length of raw_data: "+str(len(byte_data)))
        raw_data = np.frombuffer(byte_data,dtype=np.uint8)
        image=cv2.imdecode(raw_data,cv2.IMREAD_COLOR)
        print(image.shape)
        try:
            cv2.imshow("Video",image)
        except:
            traceback.print_exc(file=sys.stdout)
    def writeFrame(self,data):
        cachename = CACHE_FILE_NAME+str(self.sessionId)+CACHE_FILE_EXT
        try:
            file = open(cachename,"wb")
        except:
            traceback.print_exc(file = sys.stdout)
        try:
            file.write(data)
        except:
            traceback.print_exc(file = sys.stdout)
        file.close()
        return cachename
    def updateVideo(self,file):
        try:
            photo = ImageTk.PhotoImage(Image.open(file))
        except:
            traceback.print_exc(file = sys.stdout)
        self.label.configure(image=photo,height=288)
        self.label.image = photo
    def connectToServer(self):
        self.rtspSocket=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.rtspSocket.connect((self.serveraddr,self.serverport))
        except:
            MessageBox.showwarning('Connection failed to Server')
            traceback.print_exc(file=sys.stdout)
    def sendRtspRequest(self,requestCode):
        if requestCode == self.SETUP and self.state == self.INIT:
            threading.Thread(target=self.recvRtspResponse).start()
            self.rtspSeq = 1
            request = "SETUP "+str(self.filename)+"\n"+str(self.rtspSeq)+"\n"+" RTSP/1.0 RTP/UDP "+str(self.rtpport)
            print("send setup request")
            self.rtspSocket.send(request.encode('utf-8'))
            self.requestSent = self.SETUP
        elif requestCode == self.PLAY and self.state == self.READY:
            #threading.Thread(target=self.recvRtspResponse).start()
            self.rtspSeq +=1
            request = "PLAY "+"\n"+str(self.rtspSeq)
            self.rtspSocket.send(request.encode('utf-8'))
            print("Play request sent to Server")
            self.requestSent = self.PLAY
        elif requestCode == self.PAUSE and self.state == self.PLAYING:
            print("Pause request")
            self.rtspSeq+=1
            request = "PAUSE "+"\n"+str(self.rtspSeq)
            self.rtspSocket.send(request.encode('utf-8'))
            self.requestSent = self.PAUSE
        elif requestCode == self.QUIT and not self.state == self.INIT:
            self.rtspSeq+=1
            request = "TEARDOWN "+"\n"+str(self.rtspSeq)
            self.rtspSocket.send(request.encode('utf-8'))
            self.requestSent = self.QUIT
        else:
            return False
    def recvRtspResponse(self):
        while(1):
            response = self.rtspSocket.recv(1024)
            if response:
                self.HandleResponse(response.decode('utf-8'))
    def HandleResponse(self,data):
        print("Parsing response tcp packet")
        if str(data) == 'BREAK':
            self.quitAcked = 1
        lines = data.split('\n')
        seqNum = int(lines[1].split(' ')[1])
        if seqNum == self.rtspSeq:
            sessionId = int(lines[2].split(' ')[1])
            if self.sessionId == 0:
                self.sessionId = sessionId
            if self.sessionId == sessionId:
                if self.sessionId == sessionId:
                    if int(lines[0].split(' ')[1]) == 200:
                        if self.requestSent == self.SETUP:
                            print("Ready")
                            self.state = self.READY
                            self.openRtpPort()
                        elif self.requestSent == self.PLAY:
                            self.state = self.PLAYING
                        elif self.requestSent == self.PAUSE:
                            print("Pause response from server")
                            self.state = self.READY
                            self.playEvent.set()
                        elif self.requestSent == self.QUIT:
                            self.quitAcked = 1
    def openRtpPort(self):
        self.rtpSocket.settimeout(0.5)
        try:
            self.rtpSocket.bind(("0.0.0.0",self.rtpport))
        except:
            traceback.print_exc(file = sys.stdout)
            MessageBox.showwarning('Bind udp port failed')
    def handler(self):
        self.pauseVideo()
        if MessageBox.askokcancel("Are you want to quit"):
            self.quit()
        else:
            threading.Thread(target = self.listenRtp).start()
            self.sendRtspRequest(self.PLAY)
