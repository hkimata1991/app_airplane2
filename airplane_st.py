# 基本ライブラリ
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFilter,ImageOps

import torch
import io
#from airplane_class_1 import transform, Net # animal.py から前処理とネットワークの定義を読み込み
#import airplane_maesyori_1


# maesyori
def keepAspectResizeSimple(path, size):
    # 画像の読み込み
    image = Image.open(path)
    # サイズを幅と高さにアンパック
    width, height = size
    # 矩形と画像の幅・高さの比率の小さい方を拡大率とする
    ratio = min(width / image.width, height / image.height)
    # 画像の幅と高さに拡大率を掛けてリサイズ後の画像サイズを算出
    resize_size = (round(ratio * image.width), round(ratio * image.height))
    # リサイズ後の画像サイズにリサイズ
    resized_image = image.resize(resize_size)
    return resized_image

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

# net
# 必要なモジュールのインポート
from torchvision import transforms
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
#学習時に使ったのと同じ学習済みモデルをインポート
from torchvision.models import efficientnet_b0 

# 学習済みモデルに合わせた前処理を追加
transform = transforms.Compose([
    transforms.Resize(224),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Grayscale(num_output_channels=3),
])

#　ネットワークの定義
class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()

        #学習時に使ったのと同じ学習済みモデルを定義
        self.feature = efficientnet_b0(pretrained=True) 
        self.bnr = nn.BatchNorm1d(1000)
        self.fc = nn.Linear(1000, 10)

    def forward(self, x):
        #学習時に使ったのと同じ順伝播
        h = self.feature(x)
        h = self.bnr(h)
        h = self.fc(h)
        return h

# 学習済みモデルをもとに推論する
def predict(img):
    # ネットワークの準備
    net = Net().cpu().eval()
    # # 学習済みモデルの重み（dog_cat.pt）を読み込み
    net.load_state_dict(torch.load('./class_5.pt', map_location=torch.device('cpu')))
    #　データの前処理
    img = transform(img)
    img =img.unsqueeze(0) # 1次元増やす
    #　推論
    # 推論の実行
    with torch.no_grad():
        y1 = net(img)
        y2 = F.softmax(y1, dim=-1)
        y = torch.argmax(y2).cpu().detach().numpy()
        y2 = y2.cpu().detach().numpy()
    #y = torch.argmax(net(img), dim=1).cpu().detach().numpy()
    return y,y2

#　推論したラベルから犬か猫かを返す関数
def getName(label):
    if label==0:
        return 'Boeing-787'
    elif label==1:
        return 'MD-90'
    elif label==2:
        return 'Airbus-A320neo' 
    elif label==3:
        return 'Boeing-777' 
    elif label==4:
        return 'Airbus-A330' 
    elif label==5:
        return 'Boeing-747'
    elif label==6:
        return 'CRJ' 
    elif label==7:
        return 'E-Jet' 
    elif label==8:
        return 'Airbus-A380' 
    elif label==9:
        return 'MSJ'  


# 生成
class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
                  nn.InstanceNorm2d(out_channels),
                  nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(Generator, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)

transforms2 = transforms.Compose([
    #Resize((256, 256), Image.BICUBIC),
    transforms.ToTensor(),
])

from torchvision.transforms.functional import to_pil_image

def tensor_to_pil(tensor):
    return to_pil_image(tensor.cpu().detach().squeeze(0))

# 重みをロード
# 学習済みモデルをもとに推論する
def predict2(img):
    # ネットワークの準備
    generator = Generator().cpu().eval()
    # # 学習済みモデルの重み（dog_cat.pt）を読み込み
    generator.load_state_dict(torch.load("./generator.pth", map_location=torch.device('cpu')))
    #　データの前処理
    img = transforms2(img)
    img =img.unsqueeze(0) # 1次元増やす
    #　推論
    # 推論の実行
    with torch.no_grad():
        fake_output = generator(img)
    #y = torch.argmax(net(img), dim=1).cpu().detach().numpy()
    return fake_output


# テキスト入力による値の動的変更
app_type = st.sidebar.selectbox(
    'モード選択 1=分類  2=生成 ',
    list(range(1, 3))
)


if app_type==1:
    # タイトルとテキストを記入
    st.title('Airplane classifiers and generators')
    st.write('飛行機の画像分類と画像生成を試してみよう！')

    st.write('その１：飛行機画像分類！')

    # file upload
    #from PIL import Image
    
    uploaded_file = st.file_uploader('Choose a image file')

    # データフレームの準備
    df = pd.DataFrame({
        #'Class' : [0, 1, 2],
        'メーカー' : ['Boeing', 'McDonnell Douglas', 'Airbus','Boeing', 'Airbus','Boeing', 'Bombardier','Embraer', 'Airbus','MHI'],
        '機種' : ['B787', 'MD-90', 'A320neo','B777','A330','B747', 'CRJ','E-Jet','A380','MSJ']
    })
    st.write('※現在の対応機種')
    # 静的なテーブル
    st.table(df)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        size = (256, 256)
        img3 = keepAspectResizeSimple(uploaded_file,size)
        img3 = expand2square(img3, (0, 0, 0))
        img3 = img3.convert('RGB')
        img_array = np.array(image)
        col1, col2 = st.columns(2)
        with col1:
            st.header("original")
            st.image(
                image, caption='upload images',
                use_column_width=True
            )
        with col2:
            st.header("resized")
            st.image(
                img3, caption='upload images',
                use_column_width=True
            )
        # 入力された画像に対して推論
        pred, pred2 = predict(img3)
        AirplaneName = getName(pred) 
        st.write('## Result')
        st.write('この飛行機はきっと',AirplaneName,'です!')
        st.write('自信度は',np.round(np.max(pred2)*100,1),'%です!')
        st.write(pred2*100)

elif app_type==2:
    # タイトルとテキストを記入
    st.title('Airplane classifiers and generators')
    st.write('飛行機の画像分類と画像生成を試してみよう！')

    st.write('その２：飛行機画像生成！')
    st.write('イラストや白黒写真からカラー写真を生成します。')
    st.write('写真や塗りつぶしがあるイラストを読み込むときは、エッジ検出をONにしてください。')
    st.write('カラー写真も読み込み可。分類ページリストの機種だと精度よく出ます。')
    edge_on = st.checkbox('エッジ検出ON')

    # file upload
    #from PIL import Image
    
    uploaded_file = st.file_uploader('Choose a image file')
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        size = (256, 256)
        img3 = keepAspectResizeSimple(uploaded_file,size)
        img3 = expand2square(img3, (255, 255, 255))
        img3 = img3.convert('L')
        img_array = np.array(image)
        if edge_on:
            img3 = img3.filter(ImageFilter.FIND_EDGES)
            img3 = ImageOps.invert(img3)
        col1, col2 = st.columns(2)
        with col1:
            st.header("original")
            st.image(
                image, caption='upload images',
                use_column_width=True
            )
        with col2:
            st.header("resized")
            st.image(
                img3, caption='upload images',
                use_column_width=True
            )
        # 入力された画像に対して推論
        pred_image = predict2(img3)
        pred_image_pil = tensor_to_pil(pred_image)
        st.write('## Result')
        st.header("生成画像！")
        st.image(
                pred_image_pil, caption='Generate_Airplane',
                use_column_width=True
            )
