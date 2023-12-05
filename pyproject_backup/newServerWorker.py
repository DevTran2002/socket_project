import random, math
import time
from random import randint
import sys, traceback, threading, socket
from VideoStream import VideoStream
from RtpPacket import RtpPacket
class ServerWorker:
        SETUP = 'SETUP'
        PLAY = 'PLAY'
        PAUSE = 'PAUSE'
        TEARDOWN = 'TEARDOWN'
        INIT = 0
        READY = 1
        PLAYING = 2
        state = INIT
        OK_200 = 0
        FILE_NOT_FOUND_404 = 1
        CON_ERR_500 = 2
        #rtsp_socket
        #client_adress
        #rtp_socket
        #session_id
        #rtp_port
        def __init__(self, clientInfo):
            (self.rtsp_socket,self.client_address) = clientInfo
            print(self.client_address)
        def run(self):
            threading.Thread(target = self.Recv_RtspRequest).start()
        def Recv_RtspRequest(self):
            while(1):
                data = self.rtsp_socket.recv(256)
                if data:
                    print("Receive rtsp message from client")
                    self.Process_RtspRequest(data.decode('utf-8'))
        def Process_RtspRequest(self,data):
            request = data.split('\n')
            line1 = request[0].split(' ')
            requestType = line1[0]
            filename = line1[1]
            seq = request[1].split(' ')
            if requestType == self.SETUP:
                if self.state == self.INIT:
                    print("Got SETUP request from client")
                    try:
                        self.Stream = VideoStream(filename)
                        self.state = self.READY
                    except IOError:
                        self.Response_Rtsp(self.FILE_NOT_FOUND_404,seq[1])
                    self.session_id = randint(100000,99999999)
                    self.Response_Rtsp(self.OK_200,seq[0])
                    self.rtp_port = request[2].split(' ')[3]
            elif requestType == self.PLAY:
                if self.state == self.READY:
                    print("Got PLAY request from client")
                    self.state = self.PLAYING
                    self.rtp_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
                    self.Response_Rtsp(self.OK_200,seq[0])
                    self.event = threading.Event()
                    self.worker = threading.Thread(target = self.sendRtp).start()
                elif self.state == PAUSE:
                    print("Got RESUME request from client")
                    self.state = self.PLAYING
            elif requestType == self.PAUSE:
                if self.state == self.PLAYING:
                    print("Got PAUSE request from client")
                    self.state = self.READY
                    self.event.set()
                    self.Response_Rtsp(self.OK_200,seq[0])
            elif requestType == self.TEARDOWN:
                print("Got TEARDOWN request from client")
                self.event.set()
                self.Response_Rtsp(self.OK_200,seq[0])
                self.rtp_socket.close()
        def sendRtp(self):
            counter = 0
            threshold = 10
            while(1):
                jit = math.floor(random.uniform(-13,5.99))
                jit = jit / 1000
                self.event.wait(0.05 + jit)
                jit = jit + 0.020
                if self.event.isSet():
                    break
                data = self.Stream.nextFrame()
                if data:
                    frameNumber = self.Stream.frameNbr()
                    try:
                        port = int(self.rtp_port)
                        prb = math.floor(random.uniform(1,100))
                        if prb > 5.0:
                                '''self.rtp_socket.sendto(self.makeRtp(data,frameNumber),(self.client_address[0],port))'''
                                self.handle_bigdata(data,64000,frameNumber)
                                counter+=1
                                time.sleep(jit)
                    except:
                        traceback.print_exc(file=sys.stdout)
                        print("Error when trying to send the RTP Packet")
        def makeRtp(self,payload,frameNbr,subseq,endmark):
                version = 2
                padding = 0
                extension = 0
                cc = 0
                marker = 0
                pt = 26 # MJPEG type
                seqnum = frameNbr
                ssrc = 0
                rtpPacket = RtpPacket()
                rtpPacket.encode(version, padding, extension, cc, seqnum, marker, pt, ssrc, payload,subseq,endmark)
                return rtpPacket.getPacket()
        def Response_Rtsp(self,code,seq):
                if code == self.OK_200:
                    reply = 'RTSP/1.0 200 OK\nCSeq: ' + seq + '\nSession: ' + str(self.session_id)
                    self.rtsp_socket.send(reply.encode('utf-8'))
                elif code == self.FILE_NOT_FOUND_404:
                    print("404 NOT FOUND")
                elif code == self.CON_ERR_500:
                    print("500 CONNECTION ERROR")
        def handle_bigdata(data,maxsize,frameNbr):
            length = int(sizeof(data)/sizeof(data[0]))
            if length>maxsize:
                size = int(length/maxsize)
                for i in range(0,size):
                    try:
                        self.rtp_socket.sendto(self.makeRtp(data[int(i*maxsize):int(i*maxsize+maxsize-1)],frameNbr,i+1,0),(self.client_address[0],port))
                    except:
                        traceback.print_exc(file=sys.stdout)
                final_length = int(length - size*maxsize)
                self.rtp_socket.sendto(self.makeRtp(data[int(size*maxsize):int(size*maxsize+final_length-1)],frameNbr,size+1,1),(self.client_address[0],port))
            else:
                try:
                    self.rtp_socket.sendto(self.makeRtp(data,frameNbr,0,1),(self.client_address[0],port))
                except:
                    traceback.print_exc(file = sys.stdout)
