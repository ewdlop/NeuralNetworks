### 整體脈絡

在「**Bayes by Backprop**」(Blundell et al., 2015) 的框架中，整個神經網路被看作一個機率模型：

* **隨機變量**：所有權重與偏差 \$w={w\_i}\$
* **先驗分佈**：\$p(w)\$（通常設為 \$\mathcal N(0,1)\$，也可用混合常態等更靈活形式）
* **似然函數**：\$p(D\mid w)=\prod\_{n=1}^N p(y\_n\mid x\_n,w)\$，對分類問題常用 softmax-cross-entropy；對回歸則是高斯似然。
* **目標**：對觀測資料 \$D={(x\_n,y\_n)}\_{n=1}^N\$ 求後驗 \$p(w\mid D)\$，並用它來做預測與量化不確定度。

但因為 \$p(w\mid D)\propto p(D\mid w),p(w)\$ 的積分無法解析計算，便採 **變分推論** 近似。

---

## 1 變分推論與 ELBO

1. **設計近似族**

   $$
     q_\theta(w)=\prod_i \mathcal N\!\bigl(w_i\mid\mu_i,\sigma_i^2\bigr),
   $$

   其中可學參數 \$\theta={\mu\_i,\rho\_i}\$，且 \$\sigma\_i=\log!\bigl(1+e^{\rho\_i}\bigr)\$ 以保證 \$\sigma\_i>0\$。

2. **最小化 KL 散度**

   $$
     \theta^\star=\arg\min_\theta \operatorname{KL}\!\bigl(q_\theta(w)\,\|\,p(w\mid D)\bigr).
   $$

3. **等價轉寫──最大化“證據下界” (ELBO)** ([en.wikipedia.org][1])

   $$
     \mathcal L(\theta)=\underset{q_\theta}{\mathbb E}\,[\log p(D\mid w)]
        -\operatorname{KL}\!\bigl(q_\theta(w)\,\|\,p(w)\bigr),
   $$

   其中

   * **第一項**是「資料擬合」(negative log-likelihood, NLL)；
   * **第二項**是「正則化」(與先驗距離)。

   有時會乘上一個 **\$\beta\$-annealing** 係數（從 0 線性/循環升到 1）以避免前期 KL 造成過強約束。

---

## 2 重參數化梯度估計

為了讓 Monte Carlo 抽樣可反向傳播，對高斯後驗可寫

$$
  w=\mu+\sigma\;\epsilon,\quad\epsilon\sim\mathcal N(0,1),
$$

梯度即可對 \$\mu,\rho\$ 直接計算（這就是「重參數化技巧」）。

---

## 3 實際 loss（程式對應）

若以小批次 size \$m\$ 訓練，程式裏常用

```text
ELBO = NLL + β * KL / m             # 每筆平均
```

對應到我們的 `elbo_loss()`。

* `NLL ≈ −log p(D_b∣w)`：由一次前向中的樣本權重 \$w\$ 計算 cross-entropy。
* `KL`：`model.kl()` 把各 BayesianLayer 的 KL 相加。

---

## 4 貝氏預測與不確定度

最終想要的是

$$
  p(y^\* \mid x^\*,D)=\int p(y^\*\!\mid x^\*,w)\,q_{\theta^\*}(w)\,dw,
$$

用 **MC 抽樣 \$S\$ 次**近似：

$$
  \hat p(y^\* \mid x^\*,D)=\frac1S\sum_{s=1}^S
     p\bigl(y^\*\!\mid x^\*,w^{(s)}\bigr),\;
  w^{(s)}\sim q_{\theta^\*}(w).
$$

* **均值** \$\mathbb E\[\hat p]\$：最終機率（拿 argmax 即預測類別）。
* **樣本方差** \$\operatorname{Var}\[\hat p]\$：認知／模型（epistemic）不確定度。
  若亦對觀測雜訊建模（例如回歸時讓輸出同時估 \$\sigma\_\text{noise}\$），再加上觀測不確定度即可得到 **總不確定度**。

---

## 5 與傳統（頻率派）網路的關係

| 項目   | 傳統 NN                           | Bayes by Backprop BNN              |
| ---- | ------------------------------- | ---------------------------------- |
| 參數估計 | \$\hat w\$ (點估計)                | \$q\_\theta(w)\$ (分佈)              |
| 損失   | NLL + L2/dropout 正則化            | **NLL + KL**（有理論意義）                |
| 推論   | 單次 forward                      | 多重抽樣→平均                            |
| 不確定度 | 無 or 啟發式 (dropout)              | **自然產生**；可拆分 epistemic / aleatoric |
| 規模   | 參數翻倍 (\$\mu,\rho\$)；計算量≈\$S\$ 倍 | 需選擇 \$S\$，可在 CPU/GPU 上仍然可行         |

---

## 6 數學要點小結

1. **貝氏公式**：\$p(w\mid D)\propto p(D\mid w),p(w)\$.
2. **變分近似**：用 \$q\_\theta(w)\$ 取代難算的真後驗。
3. **ELBO**：最大化 \$\mathbb E\_{q\_\theta}\[\log p(D|w)]-\text{KL}\$ 等價於最小化 KL。
4. **重參數化**：\$w=\mu+\sigma\epsilon\$ 使梯度可穿透抽樣步驟。
5. **KL annealing**：平衡先驗與資料；早期更像 MLE，後期回歸正規貝氏。
6. **MC 預測**：樣本平均給出機率；方差量化模型不確定度。

這一套流程正是程式碼中 `BayesianLinear`, `BayesianMLP`, `elbo_loss`, `mc_predict` 各函式/方法的數學根據。想深入延伸，可參考原始論文 *Weight Uncertainty in Neural Networks*（即 BBB）([arxiv.org][2]) 與後續 survey 2006.12024。祝你研究愉快！

[1]: https://en.wikipedia.org/wiki/Evidence_lower_bound?utm_source=chatgpt.com "Evidence lower bound"
[2]: https://arxiv.org/abs/1505.05424?utm_source=chatgpt.com "Weight Uncertainty in Neural Networks"
