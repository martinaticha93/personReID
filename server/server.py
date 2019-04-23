import sys
sys.path.append('..')
from cgan import cgan
sys.path.remove('..')
import PIL.Image as Image
from http.server import BaseHTTPRequestHandler, socketserver
import base64
from io import BytesIO
import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import time

MAX_STORED_IMAGES = 9
MODEL_PATH = "models/model_fs_19.ckpt"
PORT = 8080

class Counter():
    def __init__(self):
        self.number = 1 

class ImageProcessor():
    def __init__(self):        
        self.input_batch, _, _, _, self.outputs = cgan()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())   
        self.saver = tf.train.Saver()      
        self.saver.restore(self.sess, MODEL_PATH)        
        
    def process(self, img: np.ndarray) -> Image.Image:       
        originalWidth = img.shape[1]
        originalHeight = img.shape[0]
        img = cv.resize(img, (256, 256))        
        img = img.astype(np.float32)
        img /= 255
        
        batch_input = np.zeros((1, 256, 256, 3), dtype = np.float32)
        batch_input[0,:,:,:] = img 
        
        o=self.outputs.eval(feed_dict = {self.input_batch: batch_input}, session = self.sess)
        o*=255
    
        ret=np.asarray(o[0,:,:,:], dtype = np.uint8)
        ret=cv.resize(ret, (originalWidth, originalHeight))         
        return Image.fromarray(ret, 'RGB')    

class testHTTPServer_RequestHandler(BaseHTTPRequestHandler):   
    image_processor = ImageProcessor()    
    counter = Counter()
  
    def do_GET(self) -> None:        
        self.send_response(200)   
        
        if self.path == "/edges2car.js":           
            self.send_javascript()         
        elif self.path[1:] in os.listdir('images'):           
             self.send_image()          
        elif self.path[1:] in os.listdir('outputs'):                           
            self.send_output()        
        elif self.path[5:] in os.listdir('random_examples_results'):                      
            self.send_example_result()                   
        elif self.path[1:] in os.listdir('random_examples'):
            self.send_example()        
        else:            
            self.send_html()            
    
    def send_html(self) -> None:
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        file = open("index.html","rb")        
        file_data = file.read()
        file.close()
        self.wfile.write(file_data) 
            
    def send_javascript(self) -> None:    
        self.send_header('Content-type', 'application/javascript')
        self.end_headers()
        file = open("edges2car.js", "rb")
        file_data = file.read()
        file.close()
        self.wfile.write(file_data)
        
    def send_image(self) -> None:
        if self.path[-3:] == 'png':
            self.send_header('content-type', "image/png")
        elif self.path[-3:] == 'jpg':          
            self.send_header('content-type', "image/jpg")
        self.end_headers() 
        
        file = open(os.path.join('images', self.path[1:]), 'rb')
        file_data = file.read()
        file.close()
        self.wfile.write(file_data) 
        
    def send_output(self) -> None:
        self.send_header('content-type', "image/jpg")
        self.end_headers()            
        file = open(os.path.join('outputs', self.path[1:]), 'rb')
        file_data = file.read()
        file.close()
        self.wfile.write(file_data)   
        
    def send_example_result(self) -> None:
        self.send_header('content-type', "image/png")
        self.end_headers()                  
        file = open(os.path.join('random_examples_results/', self.path[5:]), 'rb')
        file_data = file.read()
        file.close()
        self.wfile.write(file_data) 
    
    def send_example(self) -> None:
        if not self.path[1:] in os.listdir('random_examples_results'):            
            im = plt.imread(os.path.join("random_examples/", self.path[1:]))
            plt.imsave("test_result.jpg", im)
            img = cv.imread("test_result.jpg")
            ret = self.image_processor.process(img)
            ret.save(os.path.join('random_examples_results/', self.path[1:])) 
                
        self.send_header('content-type', "image/png")
        self.end_headers()            
        file = open(os.path.join('random_examples/', self.path[1:]), 'rb')
        file_data = file.read()
        file.close()
        self.wfile.write(file_data)   
        
    def do_POST(self) -> None:        
        self.send_response(200)       
        self.send_header('Content-type', 'text/html')
        self.end_headers()                
               
        content_len = int(self.headers.get('content-length', 0))
        post_body = self.rfile.read(content_len)         
        s = str(post_body)
        im = Image.open(BytesIO(base64.b64decode(s[24:-1])))
        im.save("received.png")        
        im = plt.imread("received.png")       
        name = f"{int(time.time())}{self.counter.number}.jpg"
        plt.imsave(os.path.join("drawings/", name), im)
        img = cv.imread(os.path.join("drawings/", name))
        ret = self.image_processor.process(img)        
        if self.counter.number > MAX_STORED_IMAGES:
            for file in os.listdir('outputs'):
                os.remove(os.path.join('outputs/', file))            
            self.counter.number = 1
            
        ret.save(os.path.join('outputs/', name))        
        self.counter.number += 1       
        self.wfile.write(bytes(name, "utf8"))         
        return   


def run() -> None:
  print('starting server...')  
  httpd = socketserver.TCPServer(("", PORT), testHTTPServer_RequestHandler)
  print('running server...')
  httpd.serve_forever()
 
if __name__ == "__main__": 
    run()
