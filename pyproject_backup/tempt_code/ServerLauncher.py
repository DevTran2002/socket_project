import sys,socket,traceback
from ServerWorker import ServerWorker
class Server:
    def main(self):
        try:
            SERVER_PORT = int(sys.argv[1])
        except:
            print_exc(file=sys.stdout)
        rtspSocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        rtspSocket.bind(('',SERVER_PORT))
        while(1):
            print("Listing Message from Client ...")
            rtspSocket.listen(5)
            clientInfo = rtspSocket.accept()
            ServerWorker(clientInfo).run()
if __name__ == "__main__":
    (Server()).main()
