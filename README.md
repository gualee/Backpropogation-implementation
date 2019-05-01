# Backpropogation-implementation
### 透過Python的sklearn函式庫來實作倒傳遞類神經網路分類器

### 以下為演算法執行步驟：
1.	載入sklearn套件並讀取wine資料集
2.	將資料集做正規化處理，使各維度的範圍在0~1之間，並以各50%區分訓練集和測試集
3.	使用多層感知器分類器做樣本訓練
4.	計算出正確率並呈現結果


此程式碼是使用sklearn函式庫來實作倒傳遞類神經網路分類器，故程式碼僅不到40行。在使用函式庫之前，先載入pandas、sklearn的preprocessing、model_selection、metrics以及neural_network來使用多層感知器分類器等套件。第一步先讀取資料集，使用pandas來讀取wine資料集。第二步，進行資料預處理，在做資料正規化，有很多選項可以做，比如StandardScaler、MinMaxScaler及Normalizer。這裡選擇minmaxscaler是來將各維度數值保持在[0,1]區間，公式為x_std = (x – x.min) / ( x.max – x.min)，因為最後一個維度的數值和前面的範圍差太多。做完正規化後，使用train_test_split函式將資料集以各50%區分訓練和測試集。第三步是使用MLP分類器，此實作使用到backpropogation演算法來執行，參數設定為：激勵函式使用預設的relu，權重優化選擇使用adam的隨機梯度下降最佳化，隨機狀態設定為10，代表seed作為隨機數字生成器。參數設定完後即進行訓練fit和預測predict。第四步，將訓練結果以準確率來呈現，得到的正確率高達98%左右。
