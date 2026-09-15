def count_parameters(model, verbose=True):
    """统计PyTorch模型中的参数数量
    参考: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7.

    使用示例:
    from utils.utils import count_parameters 
    count_parameters(model)
    import sys
    sys.exit(1)
    """
    # 计算模型中所有参数的总数
    n_all = sum(p.numel() for p in model.parameters())
    # 计算模型中可训练参数的总数(requires_grad=True的参数)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 如果verbose为True,打印参数统计信息
    if verbose:
        print("Parameter Count: all {:,d}; trainable {:,d}".format(n_all, n_trainable))
    # 返回所有参数数量和可训练参数数量
    return n_all, n_trainable

