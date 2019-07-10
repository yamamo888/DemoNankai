# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:46:33 2019

@author: yu
"""
import os
import pickle
import glob
import pdb

import numpy as np
import seaborn as sns
import cv2

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from mpl_toolkits.basemap import Basemap
from natsort import natsorted

from math import modf


class Data:
    def __init__(self,dataPath,fileName,nCell=8,nYear=10000):
        
        self.dataPath = dataPath
        self.logFullPath = os.path.join(self.dataPath,fileName)
        self.yInd = 1
        self.vInds = [2,3,4,5,6,7,8,9]
        self.nCell = nCell
        self.nYear = nYear
        # しんのデータに合わせて１４００年間
        self.allYear = 1400
        # 安定
        self.stable_year = 2000
        self.yV = np.zeros([nYear,nCell])
    #--------------------------------------------------------------------------     
    def loadABLV(self):
        self.data = open(self.logFullPath).readlines()
        
        self.B = np.zeros(self.nCell)
        
        for i in np.arange(1,self.nCell+1):
            tmp = np.array(self.data[i].strip().split(",")).astype(np.float32)
            self.B[i-1] = tmp[1]
            
        # Vの開始行取得
        isRTOL = [True if self.data[i].count('value of RTOL')==1 else False for i in np.arange(len(self.data))]
        vInd = np.where(isRTOL)[0][0]+1
        
        # Vの値の取得（vInd行から最終行まで）
        flag = False
        for i in np.arange(vInd,len(self.data)):
            tmp = np.array(self.data[i].strip().split(",")).astype(np.float32)
            
            if not flag:
                y = tmp
                flag = True
            else:
                y = np.vstack([y,tmp])
        
        return y
    #--------------------------------------------------------------------------     
    def convV2YearlyData(self,V):
        # 2000年以降で開始終了インデックス取得
        syearInd = np.where(np.floor(V[:,self.yInd])==self.stable_year)[0][0]
        # 2000年以降のすべり速度
        predV = V[syearInd:,:]
        # 時間とslip velocityに分ける
        times = predV[:,self.yInd]
        sv = predV[:,self.yInd+1:]
        # 2000-100000
        year = np.arange(self.stable_year,self.nYear)
        cnt = 0
        while True:
            try:
                if not any(times.astype(int)==year[cnt]):
                    # 滑ってる年数以外に前年の累積変位を入れる
                    sv = np.concatenate((sv[:np.where(times.astype(int)==year[cnt-1])[0][0]+1],sv[np.where(times.astype(int)==year[cnt-1])][0][np.newaxis],sv[np.where(times.astype(int)==year[cnt-1])[0][0]+1:]),0)
                    # 年数は滑っていない年数を入れる
                    times = np.concatenate((times[:np.where(times.astype(int)==year[cnt-1])[0][0]+1],np.array(year[cnt])[np.newaxis],times[np.where(times.astype(int)==year[cnt-1])[0][0]+1:]),0)
                    print(year[cnt])
                    #slipVelocity = np.concatenate([slipVelocity[np.where(Times.astype(int)<year[cnt])],non_slipVelocity[np.newaxis],slipVelocity[np.where(Times.astype(int)>year[cnt])]],0)
                    #Times = np.concatenate([Times[np.where(Times.astype(int)<year[cnt])],year[cnt][np.newaxis],Times[np.where(Times.astype(int)>year[cnt])]],0)
                    cnt += 1 
                else:
                    cnt += 1    
                # 3399ですとっぷ
                if year[cnt]==year[-1]:
                        break    
            except IndexError:
                print("IndexError")
        # 初めの年は滑っていないとする
        nonslip = np.zeros(8)[np.newaxis]
        # すべり速度取得
        SV = np.concatenate((nonslip,(sv[1:] - sv[:-1])),0)
        # 最近傍開始インデックス
        minyearInd = 6368
        sInd = np.where(times.astype(int)==minyearInd)[0][0]
        eInd = np.where(times.astype(int)==minyearInd+self.allYear)[0][0]
        
        # 最小誤差のすべり速度＆年数取得
        slipVelocity = SV[sInd:eInd,:]
        Times = times[sInd:eInd]
        
        return slipVelocity,Times
#--------------------------------------------------------------------------     
#--------------------------------------------------------------------------     
class Demonstrate:
    def __init__(self,dataPath,visualPath,maskPath):
        
        self.dataPath = dataPath
        self.visualPath = visualPath
        self.maskfullPath = maskPath
        self.japan = "japan"
        self.maskPath = "mask"
        self.combinePath = "combine"
        
        self.allYear = 1400
        
        self.yInd = 1
    #--------------------------------------------------------------------------     
    def Heatmap(self,slipVelocity,Times):
        # heatmap 設定
        cmap = "Reds"
            
        for i in range(slipVelocity[:5].shape[0]):
            
            plt.close()
            sns.heatmap(slipVelocity[i][np.newaxis],center=slipVelocity.mean(),cmap=cmap,cbar=False)
            plt.tick_params(labelbottom=False,labelleft=False,bottom=False,left=False)
            
            mstime,year = modf(Times[i])          
            # 保存
            self.imgfullPath = os.path.join(self.visualPath,"image_{}_{}Y_{}MS.png".format(i,year,mstime))
            plt.savefig(self.imgfullPath)
            plt.close()
            
            #----------------------------------------------------------
            # 同じ年日時のjapanmap作成
            japanmapfullPath = self.MakeJapanMap(i,year,mstime)
            # サイズ変更；拡大 japanmap var.
            resized_japanmap = self.Resize(japanmapfullPath)
            # サイズ変更：拡大　mask var.
            resized_mask = self.Resize(self.maskfullPath)
            # サイズ変更；拡大 heatmap var.
            resized_slip = self.Resize(self.imgfullPath)
            # 回転
            rotated_slip = self.Rotation(resized_slip)
            # マスク画像保存
            maskfullPath = os.path.join(self.maskPath,"Mask_{}_{}Y_{}MS.png".format(i,year,mstime))
            cv2.imwrite(maskfullPath,resized_mask)
            # Mask
            masked_slip = self.Mask(rotated_slip,maskfullPath)
            # 画像を合わせる
            self.CombineSlipVelocityJapan(i,year,mstime,resized_japanmap,masked_slip,maskfullPath)
            print(i)
    #--------------------------------------------------------------------------     
    def MakeJapanMap(self,ind,year,mtime):
        # 緯度経度指定
        west = 129
        south = 30
        east = 141
        north = 38
        
        # 日本地図作成
        japanMap = Basemap(projection='merc',resolution='h',llcrnrlon=west,llcrnrlat=south,urcrnrlon=east,urcrnrlat=north)
        japanMap.drawcoastlines(color='green')
        japanMap.drawcountries(color='palegreen')
        japanMap.fillcontinents(color='palegreen', lake_color="lightblue")
        japanMap.drawmapboundary(fill_color='lightcyan')
        
        plt.title("{}Year {}ms".format(ind,mtime),fontsize=22,color="m")
        
        japanPath = "japanMap_{}_{}Y_{}MS.png".format(ind,year,mtime)
        japanfullPath = os.path.join(self.japan,japanPath)
        plt.savefig(japanfullPath)
        plt.close()
        return japanfullPath
       
    #--------------------------------------------------------------------------
    def Resize(self,imgPath):
      
        img = cv2.imread(imgPath)
        # 指定サイズ
        h,w = 960,1280
        # 指定したサイズに変更
        img_resized = cv2.resize(img , dsize=(w,h))
        
        return img_resized   
    #--------------------------------------------------------------------------
    def Rotation(self,img):
        
        img_height,img_weight = img.shape[0],img.shape[1]
        
        # 回転角の指定
        angle = 20
        rad = np.radians(angle)
        
        sine,cosine = np.abs(np.sin(rad)),np.abs(np.cos(rad))
        tri = np.array([[cosine,sine],[sine,cosine]],np.float32)
        old_size = np.array([img_weight,img_height],np.float32)
        new_size = np.ravel(np.dot(tri,old_size.reshape(-1,1)))
        
        # 回転
        affine = cv2.getRotationMatrix2D((img_weight/2.0,img_height/2.0),angle,1.0)
        # 平行移動
        affine[:2,2] += (new_size-old_size)/2.0
        # リサイズ
        affine[:2,:] *= (old_size/new_size).reshape(-1,1)
        img_rot = cv2.warpAffine(img, affine, (img_weight,img_height), flags=cv2.INTER_CUBIC)
        
        return img_rot
    #--------------------------------------------------------------------------
    def Mask(self,img,maskfullPath):
        
        mask = cv2.imread(maskfullPath,0)
        # マスク処理
        sv_masked = cv2.bitwise_and(img,img,mask=mask)
        
        return sv_masked
        
    #--------------------------------------------------------------------------
    def CombineSlipVelocityJapan(self,ind,year,mtime,background,foreground,maskfullPath):
      
        maskimg = cv2.imread(maskfullPath,0)
        mask = 0
        #輪郭抽出
        contours, hierarchy = cv2.findContours(maskimg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(foreground)
        cv2.drawContours(mask,contours,-1,color=(255,255,255),thickness=-1)
        
        # 合成
        h,w = foreground.shape[0],foreground.shape[1]
        # 前景画像を張り付ける位置
        x,y = 10,10
        # 場所指定？？
        rocation = background[:y+h,:x+w,:]
        img_combined = np.where(mask==255,foreground,rocation)
        
        # 保存
        combinefullPath = os.path.join(self.combinePath,"Combine_{}_{}Y_{}MS.png".format(ind,year,mtime))
        cv2.imwrite(combinefullPath,img_combined)
        
    #--------------------------------------------------------------------------     
    def Animation(self,nankaimap):
        
        cap = cv2.VideoCapture(0)
        
        img_width,img_height = 640,360
        
        fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
        video  = cv2.VideoWriter('ImgVideo.mp4', fourcc, 20.0, (img_width, img_height))
        
        # 辞書順->自然順
        nankaimap_list = []
        for path in natsorted(nankaimap):
            print(path)
            nankaimap_list.append(path)
        
        # 再生画像用ストックリスト
        for i in range(len(nankaimap_list)):
            img = cv2.imread(nankaimap_list[i])
            img = cv2.resize(img,(img_width, img_height))
            video.write(img)
        
        # 保存
        video.release()
        cap.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    
    isWindow = True
    
    dataPath = "./demonstration"
    visualPath = "./visualization"
    animationPath = "combine"
    logsPath = "*.txt"
    imgPath = "*.png"
    nankaifilePath = "nankairireki.pkl"
    
    filePath = os.path.join(dataPath,logsPath)
    nakaiFilePath = os.path.join(dataPath,nankaifilePath)
    # 南海トラフの白黒画像
    maskPath = os.path.join("nankaitrough.png")
    
    files = glob.glob(filePath)
    
    # parameters
    nCell = 8
    nYear = 10000
    """
    # しんの南海トラフ履歴データ取得
    with open(nankaifilePath,"rb") as fp:
        groundtruth = pickle.load(fp)
    """
    #pdb.set_trace()
    for fID in np.arange(len(files)):
        # １つずつfaile取り出し
        if isWindow:
            file = files[fID].split("\\")[1]
        else:
            file = files[fID].split("/")[1]
            
        # 時系列データ取得
        myData = Data(dataPath,file,nCell=nCell,nYear=nYear)
        """
        predict = myData.loadABLV()
        with open("nankaiV","wb") as fp:
            pickle.dump(predict,fp)
        
        slip_velocity,times = myData.convV2YearlyData(predict)
        #-------------------------
        
        
        with open(proposed_slipvelocity,"wb") as fp:
            pickle.dump(slip_velocity,fp)
            pickle.dump(times,fp)
        """
        
        proposed_slipvelocity = os.path.join(dataPath,"slipvelocity_times.pkl")
        
        with open(proposed_slipvelocity,"rb") as fp:
            slip_velocity = pickle.load(fp)
            times = pickle.load(fp)
            
        #--------------------- デモ --------------------------------------------
        myDemo = Demonstrate(dataPath,visualPath,maskPath)
        # slipvelocity の heatmap
        #myDemo.Heatmap(slip_velocity,times)
        #-------------------- アニメーション ---------------------------------------
        demoPath = os.path.join(animationPath,imgPath)
        demofiles = glob.glob(demoPath)
    
        myDemo.Animation(demofiles)
            