# YOLOv5 模型訓練與結果分析

## 📌 實驗目的
透過 YOLOv5 模型進行影像偵測訓練，了解模型的訓練過程、評估指標與結果視覺化方法。

參考:https://blog.csdn.net/qq_51070956/article/details/134503497

## 🪄 第一步：將硬體加速器改為 GPU

在執行 YOLOv5 訓練前，請先確認 Colab 的運算環境設定為 **GPU**，以加速模型訓練。

步驟如下：
1. 點選上方選單「**執行階段 (Runtime)**」  
2. 選擇「**變更執行階段類型 (Change runtime type)**」  
3. 在「硬體加速器 (Hardware accelerator)」中選擇 **GPU**  
4. 點擊「**儲存 (Save)**」

確認後，Colab 會自動重啟運算環境。

<img width="790" height="715" alt="step0" src="https://github.com/user-attachments/assets/5636f5a4-c7df-4923-be9a-12945a8651c2" />

## 💾 第二步：掛接自己的雲端硬碟

若希望將訓練資料或模型權重儲存在 Google 雲端硬碟中，  
可先掛載雲端硬碟到 Colab：

```python
from google.colab import drive
drive.mount('/content/drive')
```
點選連線至Google雲端硬碟
<img width="787" height="275" alt="2" src="https://github.com/user-attachments/assets/6aa856cb-a56e-486e-ab3b-ac48eb069fb5" />

這樣就是掛接好了!

<img width="472" height="163" alt="2_1" src="https://github.com/user-attachments/assets/412f406c-9449-4a63-a6d8-7ad36490ea42" />

## ⚙️ 第三步：環境設定
```python
# 切換到 /content
%cd /content
```
<img width="466" height="135" alt="3" src="https://github.com/user-attachments/assets/5077fc27-8ad7-4f15-838f-b0c359323805" />

```python
# 從 GitHub 下載 YOLOv5 專案
!git clone https://github.com/ultralytics/yolov5.git
```
<img width="1442" height="278" alt="3_1" src="https://github.com/user-attachments/assets/232d3db3-4c68-4c96-903f-3f8547a43467" />

```python
# 進入 YOLOv5 資料夾
%cd yolov5
```
<img width="465" height="135" alt="3_2" src="https://github.com/user-attachments/assets/3ca38103-1a9d-4f68-bd98-20a383d20321" />


```python
# 安裝相依套件
!pip install -r requirements.txt
```
<img width="1692" height="700" alt="3_3" src="https://github.com/user-attachments/assets/d7c81c00-ca87-492c-98fe-d089814ec034" />

<img width="1689" height="702" alt="3_4" src="https://github.com/user-attachments/assets/236ba6c2-00be-4561-91ba-0a14a07b43fa" />

## 🧠 第四步：模型訓練
執行 YOLOv5 專案中的 train.py 腳本（訓練程式），並且加上 --rect 參數。
```python
!python train.py --rect
```
要輸入3，然後按enter
<img width="1767" height="494" alt="螢幕擷取畫面 2025-10-29 120625" src="https://github.com/user-attachments/assets/d64481f1-38c4-4155-9307-62d2f73c11f2" />

大概跑了6、7分鐘
可以看到最下面存至runs/train/exp
<img width="1073" height="692" alt="螢幕擷取畫面 2025-10-29 121355" src="https://github.com/user-attachments/assets/8b6764c2-52c2-4084-bc65-b03f41ff6a36" />

## 📈 第五步：使用 TensorBoard 觀察訓練過程與視覺化分析

```python
%load_ext tensorboard
```
<img width="406" height="96" alt="5" src="https://github.com/user-attachments/assets/4a236c5f-6eac-48cb-8a1d-63fd586a2bab" />

```python
%tensorboard --logdir runs/train/exp
```

###  📊 YOLOv5 評估圖解釋


1. **F1-score** 是 Precision 和 Recall 的 **「調和平均」**

這張圖會顯示在不同閾值下 F1-score 的變化。

📘 看法：

*通常會有一個峰值（最佳點），代表在那個 threshold 下 Precision 和 Recall 取得最佳平衡。*

*YOLOv5 在評估時也會用這個點來自動挑選最佳模型閾值。*

<img width="1793" height="743" alt="5_1" src="https://github.com/user-attachments/assets/45a9caed-f845-430f-ad76-8e840d4efc1c" />

2. **PR_curve (Precision-Recall Curve)**

橫軸是 Recall、縱軸是 Precision。

顯示在不同 threshold 下 Precision 與 Recall 的平衡。

常用來看模型在 **「找得多 vs 找得準」** 之間的取捨。

📘 看法：

*曲線越往右上角貼（面積越大），代表模型整體表現越好。*

*這張圖的面積（AUC）其實就是 mAP (mean Average Precision) 的來源之一。*

<img width="1779" height="731" alt="5_2" src="https://github.com/user-attachments/assets/e8ae2dea-7620-4c0a-9c27-1d8378a9237c" />

3. **P_curve (Precision Curve)**

P = Precision（精確率）：預測為「正確」的比例。

曲線顯示 在不同置信度閾值 (confidence threshold) 下，模型的精確率變化。

意思是：如果只保留模型最有把握的預測（高信心值），Precision 通常會上升。

📘 看法：

*曲線越高、越平穩，代表模型預測的準確率越高。*

<img width="1787" height="643" alt="5_3" src="https://github.com/user-attachments/assets/bed8a814-12af-49a1-afec-df54e7e409f7" />

4. **R_curve (Recall Curve)**

R = Recall（召回率）：真正的正樣本被模型找到的比例。

顯示在不同置信度閾值下，模型能「找出」多少實際存在的目標。

📘 看法：

*曲線越高，表示模型偵測漏掉的越少（越能找到全部目標）。*

<img width="1796" height="694" alt="5_4" src="https://github.com/user-attachments/assets/ccbe02b7-9bfa-4998-86f7-0729f6c20737" />

5. **confusion_matrix (混淆矩陣)**

是最直觀的「分類錯誤分析圖」。

直行代表 真實類別 (True label)，橫行代表 模型預測類別 (Predicted label)。

📘 看法：

*對角線（左上→右下）越亮越好，代表模型預測正確。*

*如果某些格子偏亮且不在對角線上，表示模型常把那兩個類別混淆。*

<img width="1781" height="726" alt="5_5" src="https://github.com/user-attachments/assets/1ba994c0-a68e-45e2-bd87-fc30a0bd5fc7" />

## 💬 心得

這次透過 YOLOv5 的實作，了解了深度學習中物件偵測的完整流程，
從 GPU 設定、掛接雲端硬碟、模型訓練到 TensorBoard 視覺化都有實際操作。
透過觀察 P_curve、R_curve、PR_curve、F1_curve 與 confusion_matrix，
能更清楚模型在哪些類別表現較好、哪些容易混淆，
未來可進一步嘗試調整訓練參數或資料集以提升準確率。




