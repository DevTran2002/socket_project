Fp__author__ = 'Tibbers'
import sys, socket

from ServerWorker import ServerWorker

class Server:

	def main(self):
                        try:
                                SERVER_PORT = int(sys.argv[1])  # lay gia tri port duoc truyen vao
                        except:
                                print("Please add the port of protocol on the first argument of the program\n") # neu khong nhan duoc gia tri port tu chuong trinh
                        while(1):
                                rtspSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # tao socket kieu dia chi ipv4, giao thuc tcp cho 
                                rtspSocket.bind(('', SERVER_PORT))
                                print("Listing RTSP Message from Client ....")
                                rtspSocket.listen(5)  # tcp socket san sang cho toi da 5 ket noi
                                # Receive client info (address,port) through RTSP/TCP session
                                #while True
                                clientInfo = rtspSocket.accept()  # return (socket object, client address)
                                ServerWorker(clientInfo).run()

# Program Start Point
if __name__ == "__main__":
	(Server()).main()


