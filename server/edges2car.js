var canvasInput = document.getElementById("canvasInput");  
canvasInput.style.backgroundColor = '#FFF'; 
var ctxInput = canvasInput.getContext("2d");

canvasInput.onmousedown = handleMouseDown;
canvasInput.onmouseup = handleMouseUp;
canvasInput.onmousemove = handleMouseMove;

var canvasOutput = document.getElementById("canvasOutput"); 
canvasOutput.style.backgroundColor = '#DDD'; 
var ctxOutput = canvasOutput.getContext("2d");

var btnTranslate = document.getElementById("btnTranslate");   
var btnEraseAll = document.getElementById("btnEraseAll"); 
var btnErase = document.getElementById("btnErase");
var btnPencil = document.getElementById("btnPencil"); 
var btnRandom = document.getElementById("btnRandom");  

//declaring some global variables:
lastClickTime = 0 //remember time of last POST request to prevent user from spamming the server
var lastX = 0; //for drawing on canvasInput
var lastY = 0;
var hold = false; //if user is drawing on canvasInput
var eraser = false; //if users is using eraser or pencil
canvasInput.style.cursor = "url(pencil_icon.png), auto";

btnTranslate.addEventListener ("click", function() {    
    if(Date.now() - lastClickTime < 1500){
        return;
    }
    lastClickTime = Date.now();
    var img  = canvasInput.toDataURL("image/png");
    var req = new XMLHttpRequest();
    req.open("POST", "", true);
    req.setRequestHeader("Content-Type", "image/png");
    req.onload = function (oEvent) {    
      var outputImage = new Image();  
      outputImage.src = "./" + req.responseText;
      
      outputImage.onload = function () {        
        ctxOutput.drawImage(outputImage, 0, 0, canvasOutput.width, canvasOutput.height);
      }     
    }
    var blob = new Blob([img]);
    req.send(blob);
  })

btnEraseAll.addEventListener ("click", function() {
    ctxInput.clearRect(0, 0, canvasInput.height, canvasInput.width);
})

btnErase.addEventListener ("click", function() {
    eraser = true;
    canvasInput.style.cursor = "url(eraser_icon.png), auto";
})
        
btnPencil.addEventListener ("click", function() {
    eraser = false;
    canvasInput.style.cursor = "url(pencil_icon.png), auto";
})
    
btnRandom.addEventListener ("click", function() {
    var rnd = Math.floor(Math.random() * 9) + 1;
    var example = new Image();  
    example.src = "./" + rnd + ".png";      
    example.onload = function () {        
          ctxInput.drawImage(example, 0, 0, canvasInput.width, canvasInput.height);
          var exampleResult = new Image();  
          exampleResult.src = "./res_"+rnd+".png";
          exampleResult.onload = function (){
          ctxOutput.drawImage(exampleResult, 0, 0, canvasOutput.width, canvasOutput.height);
          }
    } 
})

function handleMouseDown(event) {
   hold = true;
   var rect = canvasInput.getBoundingClientRect();
   lastX = event.clientX - rect.left;
   lastY = event.clientY - rect.top; 
}

function handleMouseUp(event) {   
    hold = false;
}
  
function handleMouseMove(event){

    if (hold && eraser === false) {
      var rect = canvasInput.getBoundingClientRect();
      mouseX = event.clientX - rect.left;
      mouseY = event.clientY - rect.top; 

      ctxInput.lineWidth = 3;
      ctxInput.beginPath();
      ctxInput.moveTo(lastX, lastY);
      ctxInput.lineTo(mouseX, mouseY);
      ctxInput.stroke();

      lastX = mouseX;
      lastY = mouseY;
    } else if (hold && eraser === true) {
      var rect = canvasInput.getBoundingClientRect();
      mouseX = event.clientX - rect.left;
      mouseY = event.clientY - rect.top; 

      ctxInput.clearRect(mouseX - 15, mouseY - 15, 30, 30);   

      lastX = mouseX;
      lastY = mouseY;
   }
}





