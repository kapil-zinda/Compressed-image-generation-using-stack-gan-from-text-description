# pylab inline
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from pylab import *
import scipy.misc
from scipy.fftpack import dct,idct

from skimage.color import rgb2ycbcr,ycbcr2rgb
from PIL import Image

import random
import huffman
from collections import Counter

QYt=array([[16,11,10,16,24,40,51,61],
     [12,12,14,19,26,58,60,55],
     [14,13,16,24,40,57,69,56],
     [14,17,22,29,51,87,80,62],
     [18,22,37,56,68,109,103,77],
     [24,35,55,64,81,104,113,92],
     [49,64,78,87,103,121,120,101],
     [72,92,95,98,112,100,103,99]])
QCt=array([[17,18,24,47,99,99,99,99],
     [18,21,26,66,99,99,99,99],
     [24,26,56,99,99,99,99,99],
     [47,66,99,99,99,99,99,99],
     [99,99,99,99,99,99,99,99],
     [99,99,99,99,99,99,99,99],
     [99,99,99,99,99,99,99,99],
     [99,99,99,99,99,99,99,99]])

# Transform the image into blocks
def toBlocks(img, yLen,xLen,h,w):
  blocks = zeros((yLen,xLen,h,w,3),dtype=int16)
  for y in range(yLen):
    for x in range(xLen):
      blocks[y][x]=img[y*h:(y+1)*h,x*w:(x+1)*w]
  return array(blocks)

def myYcbcr2rgb(ycbcr):
  return (ycbcr2rgb(ycbcr).clip(0,1)*255).astype(uint8)

# operate DCT on all the blocks respectively, check the difference between step-2 and step-3
def dctOrDedctAllBlocks(blocks,yLen,xLen,h,w,type="dct"):
  f=dct if type=="dct" else idct
  dedctBlocks = zeros((yLen,xLen,h,w,3))
  for y in range(yLen):
    for x in range(xLen):
      d = zeros((h,w,3))
      for i in range(3):
        block=blocks[y][x][:,:,i]
        d[:,:,i]=f(f(block.T, norm = 'ortho').T, norm = 'ortho')
        if (type!="dct"):
          d=d.round().astype(int16)
      dedctBlocks[y][x]=d
  return dedctBlocks

def blocks2img(blocks,yLen,xLen,h,w):
  W=xLen*w
  H=yLen*h
  img = zeros((H,W,3))
  for y in range(yLen):
    for x in range(xLen):
      img[y*h:y*h+h,x*w:x*w+w]=blocks[y][x]
  return img

def image_compress(image_in):
  # pylab inline
  img = image_in
  w=8
  h=w
  xLen = img.size[1]//w
  yLen = img.size[0]//h
  runBits=1 #modify it if you want
  bitBits=3  #modify it if you want
  rbBits=runBits+bitBits ##(run,bitSize of coefficient)
  useHuffman=True #modify it if you want


  # Transfer RGB image to YCbCr Image
  img=rgb2ycbcr(img)

  blocks = toBlocks(img, yLen,xLen,h,w)

  dctBlocks=dctOrDedctAllBlocks(blocks,yLen,xLen,h,w,"dct")

  QY=QYt[:w,:h]
  QC=QCt[:w,:h]
  qDctBlocks=copy(dctBlocks)
  Q3 = moveaxis(array([QY]+[QC]+[QC]),0,2)#all r-g-b/Y-Cb-Cr 3 channels need to be quantized
  Q3=Q3*((11-w)/3)
  qDctBlocks=(qDctBlocks/Q3).round().astype('int16')




  # Zig-Zag process
  def zigZag(block):
    lines=[[] for i in range(h+w-1)] 
    for y in range(h): 
      for x in range(w): 
        i=y+x 
        if(i%2 ==0): 
            lines[i].insert(0,block[y][x]) 
        else:  
            lines[i].append(block[y][x]) 
    return array([coefficient for line in lines for coefficient in line])




  # Huffman Coding
  def huffmanCounter(zigZagArr):
    rbCount=[]
    run=0
    for AC in zigZagArr[1:]:
      if(AC!=0):
        AC=max(AC,1-2**(2**bitBits-1)) if AC<0 else min(AC,2**(2**bitBits-1)-1)
        if(run>2**runBits-1):
          runGap=2**runBits
          k=run//runGap
          for i in range(k):
            rbCount.append('1'*runBits+'0'*bitBits)
          run-=k*runGap
        run=min(run,2**runBits-1) 
        bitSize=min(int(ceil(log(abs(AC)+0.000000001)/log(2))),2**bitBits-1)
        rbCount.append(format(run<<bitBits|bitSize,'0'+str(rbBits)+'b'))
        run=0
      else:
        run+=1
    rbCount.append("0"*(rbBits))
    return Counter(rbCount)


  def runLength(zigZagArr,lastDC,hfm=None):
    rlc=[]
    run=0
    newDC=min(zigZagArr[0],2**(2**bitBits-1))
    DC=newDC-lastDC
    bitSize=max(0,min(int(ceil(log(abs(DC)+0.000000001)/log(2))),2**bitBits-1))
    code=format(bitSize, '0'+str(bitBits)+'b')
    if (bitSize>0):
      code+=(format(DC,"b") if DC>0 else ''.join([str((int(b)^1)) for b in format(abs(DC),"b")]))
    for AC in zigZagArr[1:]:
      if(AC!=0):
        AC=max(AC,1-2**(2**bitBits-1)) if AC<0 else min(AC,2**(2**bitBits-1)-1)
        if(run>2**runBits-1):
          runGap=2**runBits
          k=run//runGap
          for i in range(k):
            code+=('1'*runBits+'0'*bitBits)if hfm == None else  hfm['1'*runBits+'0'*bitBits]#end
          run-=k*runGap
        run=min(run,2**runBits-1) 
        bitSize=min(int(ceil(log(abs(AC)+0.000000001)/log(2))),2**bitBits-1)
        rb=format(run<<bitBits|bitSize,'0'+str(rbBits)+'b') if hfm == None else hfm[format(run<<bitBits|bitSize,'0'+str(rbBits)+'b')]
        code+=rb+(format(AC,"b") if AC>=0 else ''.join([str((int(b)^1)) for b in format(abs(AC),"b")]))
        run=0
      else:
        run+=1
    code+="0"*(rbBits) if hfm == None else  hfm["0"*(rbBits)]#end
    return code,newDC



  # Save the result of Run-Length to bytes
  def runLength2bytes(code):
    return bytes([len(code)%8]+[int(code[i:i+8],2) for i in range(0, len(code), 8)])



  # Saving as jpeg, do step-5 for all blocks of the image
  def huffmanCounterWholeImg(blocks):
    rbCount=zeros(xLen*yLen*3,dtype=Counter)
    zz=zeros(xLen*yLen*3,dtype=object)
    for y in range(yLen):
      for x in range(xLen):
        for i in range(3):
          zz[y*xLen*3+x*3+i]=zigZag(blocks[y, x,:,:,i])
          rbCount[y*xLen*3+x*3+i]=huffmanCounter(zz[y*xLen*3+x*3+i])
    return sum(rbCount),zz


  def savingQuantizedDctBlocks(blocks):
    rbCount,zigZag=huffmanCounterWholeImg(blocks)
    hfm=huffman.codebook(rbCount.items())
    sortedHfm=[[hfm[i[0]],i[0]] for i in rbCount.most_common()]
    code=""
    DC=0
    for y in range(yLen):
      for x in range(xLen):
        for i in range(3):
          codeNew,DC=runLength(zigZag[y*xLen*3+x*3+i],DC,hfm if useHuffman else None)
          code+=codeNew
    savedImg=runLength2bytes(code)
    return bytes([int(format(xLen,'012b')[:8],2),int(format(xLen,'012b')[8:]+format(yLen,'012b')[:4],2),int(format(yLen,'012b')[4:],2)])+savedImg,sortedHfm
  savedImg,sortedHfmForDecode=savingQuantizedDctBlocks(qDctBlocks)
  save = open("img.bin", "wb")
  save.write(savedImg)
  save.close()

  # Run-Length decoding
  def bytes2runLength(bytes):
    return "".join([format(i,'08b') for i in list(bytes)][1:-1 if list(bytes)[-1]!=0 else None])+(format(list(bytes)[-1],'0'+str(list(bytes)[0])+'b')if list(bytes)[-1]!=0 else"")

  # Zig-Zag decoding
  gaps=[i for i in range(1,8)]+[8-i for i in range(8)]+[-1]
  locations=[[int(sum(range(gaps[i-1]+1))),sum(range(gaps[i]+1))] if gaps[i]>gaps[i-1]  else [64-sum(range(gaps[i-1])),64-sum(range(gaps[i]))] for i in range(len(gaps)-1)]
  def deZigZag(zigZagArr):
    zigZagArr=[zigZagArr[l[0]:l[1]] for l in locations]
    block=zeros((h,w),dtype=int16)
    for y in range(h): 
      for x in range(w): 
        i=y+x 
        if(i%2 != 0): 
          block[y][x]=zigZagArr[i][0]
          zigZagArr[i]=zigZagArr[i][1:]
        else: 
          block[y][x]=zigZagArr[i][-1:][0]
          zigZagArr[i]=zigZagArr[i][:-1]
    return block


  # Another Zig-Zag Decoding
  if(w%2==0):
    x=concatenate([append(arange(0,i),arange(0,i)[::-1][1:]) for i in range(2,w+1,2)])
    x=append(x,(w-1-x[::-1])[w:])
    y=concatenate([append(arange(0,i),arange(0,i)[::-1][1:]) for i in range(1,w+1,2)])
    y=append(append(y,arange(0,h)),(h-1-y[::-1]))
  else:
    x=concatenate([append(arange(0,i),arange(0,i)[::-1][1:]) for i in range(2,w,2)])
    x=append(append(x,arange(0,w)),w-1-x[::-1])
    y=concatenate([append(arange(0,i),arange(0,i)[::-1][1:]) for i in range(1,w+2,2)])
    y=append(y[:-w],w-1-y[::-1])
  zzLine=dstack([x]+[y])[0]
  zzMatrix=zeros((w,h),dtype=int8)
  for i in range(len(zzLine)):
    zzMatrix[zzLine[i][1]][zzLine[i][0]]=i

  def deZigZag2(zigZagArr):
    block=zeros((h,w),dtype=int16)
    for i in range(len(zzLine)):
      block[zzLine[i][1],zzLine[i][0]]=zigZagArr[i]
    return block



  # Loading the jpeg, decompression of compressed(encoded) image
  load = open("img.bin", "rb")
  loadedbytes = load.read()
  code=bytes2runLength(loadedbytes[2:])

  def loadingQuantizedDctBlocks(loadedbytes,sortedHfm=None):
    runMax=2**runBits-1
    xLen=int(format(loadedbytes[0],'b')+format(loadedbytes[1],'08b')[:4],2)
    yLen=int(format(loadedbytes[1],'08b')[4:]+format(loadedbytes[2],'08b'),2)
    code=bytes2runLength(loadedbytes[3:])
    blocks = zeros((yLen,xLen,h,w,3),dtype=int16)
    lastDC=0
    rbBitsTmp=rbBits
    rbTmp=""
    cursor=0 #don't use code=code[index:] to remove readed strings when len(String) is large like 1,000,000. It will be extremely slow
    for y in range(yLen):
      for x in range(xLen):
        for i in range(3):
          zz=zeros(64)
          bitSize=int(code[cursor:cursor+bitBits],2)
          DC=code[cursor+bitBits:cursor+bitBits+bitSize]
          DC=(int(DC,2) if DC[0]=="1" else -int(''.join([str((int(b)^1)) for b in DC]),2)) if bitSize>0 else 0
          cursor+=(bitBits+bitSize)
          zz[0]=DC+lastDC
          lastDC=zz[0]
          r=1
          while(True):
            if(sortedHfm!=None):
              for ii in sortedHfm:
                if (ii[0]==code[cursor:cursor+len(ii[0])]):
                  rbTmp=ii[1]
                  rbBitsTmp=len(ii[0])
                  break
              run=int(rbTmp[:runBits],2)
              bitSize=int(rbTmp[runBits:],2)
            else:
              run=int(code[cursor:cursor+runBits],2)
              bitSize=int(code[cursor+runBits:cursor+rbBitsTmp],2)
            if (bitSize==0):
              cursor+=rbBitsTmp
              if(run==runMax):
                r+=(run+1)
                continue
              else:
                break
            coefficient=code[cursor+rbBitsTmp:cursor+rbBitsTmp+bitSize]
            if(coefficient[0]=="0"):
              coefficient=-int(''.join([str((int(b)^1)) for b in coefficient]),2)
            else:
              coefficient=int(coefficient,2)
            zz[r+run]=coefficient
            r+=(run+1)
            cursor+=rbBitsTmp+bitSize    
          blocks[y,x,...,i]=deZigZag2(zz)
    return blocks
  loadedBlocks=loadingQuantizedDctBlocks(loadedbytes,sortedHfmForDecode if useHuffman else None)

  incorrect=qDctBlocks.size-count_nonzero((qDctBlocks==loadedBlocks)==True)
  print("Incorrect count due to bit size of [run("+str(runBits)+"bits),bitSize("+str(bitBits)+"bits)]: \n"+str(incorrect)+" out of "+str(qDctBlocks.size))

  loadedDctImg=blocks2img(loadedBlocks,yLen,xLen,h,w)

  deDctLoadedBlocks=dctOrDedctAllBlocks(loadedBlocks*Q3,yLen,xLen,h,w, "idct")
  loadedImg=blocks2img(deDctLoadedBlocks,yLen,xLen,h,w)

  return loadedDctImg.astype(uint8), myYcbcr2rgb(loadedImg)