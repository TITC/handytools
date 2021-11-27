import math
from tqdm.std import tqdm
from cutword import pku, jieba_tools
from interface import access


def seg_sentences(txts: list, seg_type: str = "ddp", grain="Coarse", batch_size=20):
    """指定分词器分词

    Args:
        txts (list): each element in the list is a sentence
        seg_type (str, optional): denote the segmentor for a sentence.
                                 as so far, optional contains support jieba,baidu, and pku. 
                                 Defaults to "jieba".
    Returns:
        [type]: [description]
    """
    run_dict = {"pku": pku.run,
                "jieba": jieba_tools.run,
                "baidu": access.lac_batch,
                "ddp": access.grained_cut,
                "ddpseg": access.ddpseg_batch}
    if seg_type != "ddp":
        run = run_dict[seg_type]
    else:
        def grain_run(word):
            return run_dict[seg_type](word, grain)
        run = grain_run
    result = []
    for batch_idx in tqdm(range(math.ceil(len(txts)/batch_size)), desc="segmenting..."):
        cur_batch = txts[batch_idx*batch_size:(batch_idx+1)*batch_size]
        result.extend(run(cur_batch))

    return result


if __name__ == '__main__':
    txts = """(回归神经网络)，LSTM神经网络，XGBoost算法，lightGBM算法
LSTM神经网络、XGBoost算法、lightGBM算法
RNN(递归神经网络),LSTM神经网络,XGBoost算法,lightGBM算法
小波变换、复杂神经网络、故障诊断
血小板变换，复杂神经网络，故障诊断
小波变换，复杂神经网络，故障诊断
小波变换,复杂的神经网络,故障诊断
RNN是一种神经网络,包括一个或多个反馈循环。有不同的形式反馈应用于神经网络,导致不同的结构ofRNN。学习算法,梯度下降方法得到了广泛的应用。1989年,威廉姆斯和拉链介绍了实时反复学习算法(RTRL)[21]。Theliterature[22]给出了确定性的梯度下降算法收敛RNN一个神经元。本文继续forRNN梯度学习算法的理论研究。
RNN是一种包含一个或多个反馈环的神经网络。由于反馈在神经网络中的应用形式不同，导致rnn的结构也不同。在学习算法中，梯度下降法得到了广泛的应用。1989年，Williams和Zipser提出了实时循环学习算法(RTRL)[21]。文献[22]给出了单神经元RNN的确定收敛性梯度下降算法。本文对rnn的梯度学习算法进行了理论研究。
是一种包含一个或多个反馈回路的神经网络。对神经网络进行反馈，得到不同的神经网络结构。20学习算法、梯度下降法已经被广泛应用。201989年，Williams和Zipser引入了实时回归学习算法(RTRL)[21]。[22]给出了一个神经元的梯度下降算法的确定性收敛性。本文继续对梯度学习算法进行理论研究。
是一种包括一个或多个反馈回路神经网络。对神经网络应用反馈的形式不同，导致神经网络结构的不同。20学习算法、梯度下降法得到了广泛的应用。201989年，Williams和Zipser引入了实时循环学习算法(RTRL)[21]。文献[22]给出了一个神经元的梯度下降算法的确定性收敛性。本文继续对梯度学习算法的理论研究。
当使用模糊逻辑控制,它不需要一个精确的数学模型的控制系统,但语言变量来描述它。神经网络有许多优点,如非线性映射,自学,并行处理。所以模糊控制和神经网络的结合可以使隶属度函数和模糊规则可以转化为模糊神经网络。然后把分散的知识体系形成凭不断调整神经网络的重量,隶属函数更精确。
当模糊逻辑用于控制时，它不需要被控系统的精确数学模型，而需要语言变量来描述它。神经网络具有非线性映射、自学习、并行处理等优点。因此，将模糊控制与神经网络相结合，可以将隶属函数和模糊规则转化为模糊神经网络。通过不断调整神经网络的权值，使隶属函数更精确，从而形成分散的知识系统。
控制采用n模糊逻辑，控制系统不需要精确的数学模型，而需要语言变量来描述。神经网络具有非线性映射、自学习、并行处理等优点。将模糊控制与神经网络相结合，使隶属函数和模糊规则转化为模糊神经网络成为可能。通过不断调整神经网络的权值，使隶属函数更精确。""".split("\n")
    print(seg_sentences(txts=txts, seg_type="ddpseg", batch_size=20)
          )
