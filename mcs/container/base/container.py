import torch
import numpy as np

class Container:
    @staticmethod
    def load_network(network, path):
        network.load_state_dict(
            torch.load(path, map_location=lambda storage, loc: storage)
        )
        return network

    @staticmethod
    def load_optim(optimizer, path):
        optimizer.load_state_dict(
            torch.load(path, map_location=lambda storage, loc: storage)
        )
        return optimizer

    @staticmethod
    def init_next_save(initial_step_count, epoch_len):
        next_save = 0
        if initial_step_count > 0:
            while next_save <= initial_step_count:
                next_save += epoch_len
        return next_save

    @staticmethod
    def count_parameters(net):
        return sum(p.numel() for p in net.parameters() if p.requires_grad)

    @staticmethod
    def write_summaries(
            writer, step_count, total_loss, loss_dict, metric_dict, n_params
    ):
        writer.add_scalar("loss/total_loss", total_loss.item(), step_count)
        for l_name, loss in loss_dict.items():
            writer.add_scalar("loss/" + l_name, loss.item(), step_count)
        for m_name, metric in metric_dict.items():
            if m_name == "importance":
                writer.add_scalar("metric/" + m_name, metric.item(), step_count)
            elif m_name == "action":
                writer.add_histogram("metric/" + m_name, getData(metric), step_count)
            else:
                writer.add_scalar("metric/" + m_name, metric.item(), step_count)
        for p_name, param in n_params:
            p_name = p_name.replace(".", "/")
            writer.add_scalar(p_name, torch.norm(param).item(), step_count)
            if param.grad is not None:
                writer.add_scalar(
                    p_name + ".grad", torch.norm(param.grad).item(), step_count
                )



def getData(data):
    result = np.linspace(0,26,27)
    repeat = []
    for i in range(27):
        repeat.append(int(data[i]*100))
    return result.repeat(repeat)

