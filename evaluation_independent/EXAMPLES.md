# 固定规则抽取的六个例子

每个领域按 SHA256(`examples20261103:` + 样本 ID) 排序取首条，不根据对错筛选。这里只展示冻结的 greedy16 单字母答题协议输出；截断的描述文字不用于评价完整开放式问答质量。这不是新增人工标注集。

## mmstar-0050 · coarse perception

![mmstar-0050](examples/mmstar-0050.jpg)

which image is more colorful?
Options: A: The first image, B: The second image, C: nan, D: nan

数据集标准选项：**A**

**initial**（A–D 概率诊断：B；达到长度上限：True）

```text
<think>

</think>

Here is the step-by-step process
```

**pretrain**（A–D 概率诊断：B；达到长度上限：True）

```text
The image is a two-dimensional football game,
```

**sft**（A–D 概率诊断：B；达到长度上限：True）

```text
The image clearly shows two separate scenes, one in
```

**official**（A–D 概率诊断：B；达到长度上限：True）

```text
The image most closely aligns with option (A):
```

## mmstar-0449 · fine-grained perception

![mmstar-0449](examples/mmstar-0449.jpg)

Where is the table in relation to the balcony?
Options: A: On the right side of the balcony, B: In the center of the balcony, C: On the left side of the balcony, D: Cannot be determined from the information given

数据集标准选项：**B**

**initial**（A–D 概率诊断：C；达到长度上限：True）

```text
Option A: On the right side of the balcony
```

**pretrain**（A–D 概率诊断：B；达到长度上限：True）

```text
The table appears to be a casual or fantasy sc
```

**sft**（A–D 概率诊断：B；达到长度上限：True）

```text
The table in question is a balcony with a white chair
```

**official**（A–D 概率诊断：B；达到长度上限：True）

```text
The table in the balcony is located in the center of the
```

## mmstar-0737 · instance reasoning

![mmstar-0737](examples/mmstar-0737.jpg)

Based on the image, what is the significance of having two cakes on the table during the couple's celebration with their baby?
Options: A: Having two cakes allows for different cake flavors or designs for their guests., B: Having two cakes signifies that the couple is celebrating multiple occasions or milestones., C: Having two cakes indicates a preference for abundance and excess., D: Having two cakes is a common practice in most celebrations of this nature.

数据集标准选项：**B**

**initial**（A–D 概率诊断：B；达到长度上限：True）

```text
## A: Having two cakes on the table

**O
```

**pretrain**（A–D 概率诊断：B；达到长度上限：True）

```text
The cake is a cake with a baby flavor, with
```

**sft**（A–D 概率诊断：B；达到长度上限：True）

```text
The significance of having two cakes on the table during the cou
```

**official**（A–D 概率诊断：B；达到长度上限：True）

```text
The significance of having two cakes on the table during the cele
```

## mmstar-0936 · logical reasoning

![mmstar-0936](examples/mmstar-0936.jpg)

Which of these cities is marked on the map?
Options: A: Miami, B: Atlanta, C: Nashville, D: Oklahoma City

数据集标准选项：**B**

**initial**（A–D 概率诊断：B；达到长度上限：True）

```text
The correct answer is:

A: Miami, B:
```

**pretrain**（A–D 概率诊断：A；达到长度上限：True）

```text
The map is a map with a map of a map of
```

**sft**（A–D 概率诊断：B；达到长度上限：True）

```text
The map shows a close-up of a map, which
```

**official**（A–D 概率诊断：B；达到长度上限：True）

```text
The map in the image is not marked on the map. The
```

## mmstar-1013 · math

![mmstar-1013](examples/mmstar-1013.jpg)

Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
Question: In Fig. 21-25, the particles have charges $q_1=-q_2=100 \mathrm{nC}$ and $q_3=-q_4=200 \mathrm{nC}$, and distance $a=$ $5.0 \mathrm{~cm}$. What is the $x$ component of the net electrostatic force on particle 3?
Choices:
(A) 0
(B) 0.34
(C) 0.17
(D) 3

数据集标准选项：**C**

**initial**（A–D 概率诊断：D；达到长度上限：True）

```text
I need to find the net electrostatic force on particle
```

**pretrain**（A–D 概率诊断：B；达到长度上限：True）

```text
The net electrostatic force on particle 3 is given by
```

**sft**（A–D 概率诊断：B；达到长度上限：True）

```text
The net electrostatic force on particle 3 is given by
```

**official**（A–D 概率诊断：B；达到长度上限：True）

```text
To find the final value of the net electrostatic force
```

## mmstar-1443 · science & technology

![mmstar-1443](examples/mmstar-1443.jpg)

What is represented directly under the aorta?
Options: A: femoral artery, B: none of the above, C: renal vein, D: heart

数据集标准选项：**C**

**initial**（A–D 概率诊断：C；达到长度上限：True）

```text
<think>

</think>

<think>

</think>

### 一、
```

**pretrain**（A–D 概率诊断：B；达到长度上限：True）

```text
The aorta is a body of water that is a layer
```

**sft**（A–D 概率诊断：B；达到长度上限：True）

```text
The correct answer to the question is:

A: femoral
```

**official**（A–D 概率诊断：B；达到长度上限：True）

```text
The image provided does not directly correspond to the content of the question
```
