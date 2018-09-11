import torch 
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import numpy as np

class SelfAtt(nn.Module):
    def __init__(self, c):
        super(SelfAtt, self).__init__()

        self.att_f = self.conv(c, c // 4, 1, 1, bn_f=True, act=nn.ReLU)
        self.att_g = self.conv(c, c // 4, 1, 1, bn_f=True, act=nn.ReLU)
        self.att_h = self.conv(c, c, 1, 1, bn_f=True, act=nn.ReLU)

    def conv(self, cin, cout, ksize, stride=1, bn_f=True, act=nn.ReLU):
        layers = [nn.Conv2d(cin, cout, ksize, stride=stride, padding=ksize // 2)]

        if bn_f:
            layers += [nn.BatchNorm2d(cout)]
        if act is not None:
            layers += [act()]
        m = nn.Sequential(*layers)
        if torch.cuda.is_available():
            m = m.cuda()
        return m

    def hw_flatten(self, x):
        return x.view(x.size(0), x.size(1), -1)

    def forward(self, x):
        f = self.hw_flatten(self.att_f(x))
        g = self.hw_flatten(self.att_g(x))
        h = self.hw_flatten(self.att_h(x))

        s = torch.bmm(g.transpose(1, 2), f)
        beta = F.softmax(s, -1)

        o = torch.bmm(beta, h.transpose(1, 2))
        o = o.transpose(1, 2)
        o = o.view(x.size())
        return o * x

class LipNet(torch.nn.Module):
    def __init__(self, dropout_p=0.5, base=16, height=50, width=100, n_rnn=2, n_layers=1, bi=True, mode='LSTM', hidden=256):
        super(LipNet, self).__init__()
        self.base = base
        self.n_rnn = n_rnn
        self.bi = bi
        self.block1 = nn.Sequential(self.conv(3, base, 3, 1),
                                    self.conv(base, int(base * 1.5), 3, 2),
                                    self.conv(int(base * 1.5), base * 3, 3, 2))

        self.att1 = SelfAtt(base * 3)
        self.block2 = nn.Sequential(
						self.conv(base * 3, base * 4, 3, 2),
						# self.conv(base * 4, base * 4, 3, 1)
						)

        self.att2 = SelfAtt(base * 4)
        self.block3 = nn.Sequential(
						self.conv(base * 4, base * 8, 3, 2),
						# self.conv(base * 8, base * 8, 3, 1)
						)
        for i in range(4):
            if height % 2 == 1:
                height += 1
            if width % 2 == 1:
                width += 1
            height = height // 2
            width = width // 2
        self.block4 = self.fc(height * width * base * 8, base * 8)

        cell = nn.LSTM if mode == 'LSTM' else nn.GRU
        i_size = hidden if not bi else hidden * 2
        self.rnn1 = cell(input_size=base * 8, hidden_size=hidden, num_layers=n_layers, bidirectional=bi)
        self.rnn2 = cell(input_size=i_size, hidden_size=hidden, num_layers=n_layers, bidirectional=bi)

        self.FC  = self.fc(i_size, 27+1, dp=0, act=None)

        self.dropout_p  = dropout_p
        self.init()
    def conv(self, cin, cout, ksize, stride=1, bn_f=True, act=nn.ReLU):
        c_layer = nn.Conv2d(cin, cout, ksize, stride=stride, padding=ksize // 2)
        init.kaiming_normal_(c_layer.weight, nonlinearity='relu')
        init.constant_(c_layer.bias, 0)
        layers = [c_layer]

        if bn_f:
            layers += [nn.BatchNorm2d(cout)]
        if act is not None:
            layers += [act()]
        m = nn.Sequential(*layers)
        if torch.cuda.is_available():
            m = m.cuda()
        return m

    def fc(self, din, dout, dp=0.5, act=nn.ReLU):
        l_layer = nn.Linear(din, dout)
        init.kaiming_normal_(l_layer.weight, nonlinearity='sigmoid')
        init.constant_(l_layer.bias, 0)
        layers = [l_layer]
        if dp != 0:
            layers += [nn.Dropout(p=dp)]
        if act is not None:
            layers += [act()]
        m = nn.Sequential(*layers)
        if torch.cuda.is_available():
            m = m.cuda()
        return m
    def init(self):

        for m in (self.rnn1, self.rnn2):
            stdv = math.sqrt(2 / (self.base * 8 + 256))
            for i in range(0, 256 * 3, 256):
                init.uniform_(m.weight_ih_l0[i: i + 256],
                            -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0[i: i + 256])
                init.constant_(m.bias_ih_l0[i: i + 256], 0)
                if self.bi:
                    init.uniform_(m.weight_ih_l0_reverse[i: i + 256],
                                -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                    init.orthogonal_(m.weight_hh_l0_reverse[i: i + 256])
                    init.constant_(m.bias_ih_l0_reverse[i: i + 256], 0)

    def forward(self, x, input_length):
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        B, T, C, H, W = x.size()
        x = x.view(-1, C, H, W)

        x = self.block1(x)
        x = self.att1(x)

        x = self.block2(x)
        x = self.att2(x)
        x = self.block3(x)

        _, C, H, W = x.size()
        x = x.view(x.size(0), -1)
        x = self.block4(x)
        x = x.view(B, T, -1)
        x    = nn.utils.rnn.pack_padded_sequence(x, input_length, batch_first=True)
        x, h = self.rnn1(x)
        x, h = self.rnn2(x)
        x, L = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.FC(x)
        x = x.log_softmax(2)
        return x

if __name__ == '__main__':
    data = torch.randn((12, 3, 75, 50, 100))
    l = [75] * 12
    m = LipNet()
    m(data, l)