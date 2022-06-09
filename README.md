**English | 简体中文**

## Introduction
These codes are the experiments to **<font color=Blue>simulate the attack on Ring AllReduce algorithm</font>** in *Single* GPU by Pytorch.

Therefore, what you need is a **<font color=Blue>single GPU</font>** with **<font color=Blue>Pytorch</font>** available.

***<font color=Blue>BTW</font>***, the annotations in the codes will be rewritten in **<font color=Blue>English</font>** later, if I have spare time LOL.

## What's Ring AllReduce?

[Ring AllReduce介绍](https://www.jianshu.com/p/8c0e7edbefb9)

[Introduction of Ring AllReduce](https://marek.ai/allreduce-the-basis-of-multi-device-communication-for-neural-network-training.html#:~:text=Ring%20allreduce%20is%20actually%20a%20meta%20communication%20collective%2C,we%20send%20that%20message%20to%20our%20succeding%20neighbor.)

I really recommend the article posted latter(written in English one). After reading the article carefully, I believe that little problems would be occured for you to comprehend these codes.

## What's the Attack?

Briefly, the aim of attack is to make the whole training process **<font color=Blue>divergence</font>** (at least my attack is lol).

In *Attack.py* file, 2 kinds of treatments of data are provided.

## Ending

**Welcome** any questions about the codes~~~.
