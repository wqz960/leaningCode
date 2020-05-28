from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr
import math
from paddle.fluid.contrib.model_stat import summary

__all__ = ['ResNeSt50', 'ResNeSt101', 'ResNeSt200', 'ResNeSt269']

class ResNeSt():
    def __init__(self, layers, radix=1, groups=1, bottleneck_width=64, dilated=False,
                 dilation=1, deep_stem=False, stem_width=64, avg_down=False,
                 rectify_avg=False, avd=False, avd_first=False, final_drop=0.0,
                 dropblock_prob=0, last_gamma=False):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width*2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        # ResNeSt params
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first
        
        self.deep_stem = deep_stem
        self.stem_width = stem_width
        self.layers = layers
        self.dropblock_prob = dropblock_prob
        self.final_drop = final_drop
        self.dilation = dilation
        
        self.rectify_avg = rectify_avg

    def net(self, input, class_dim=1000, data_format="NCHW"):
        layers = self.layers

        conv = self.conv_bn_layer(
            input=input,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu',
            name="conv1",
            data_format=data_format)
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max',
            data_format=data_format)
        if layers >= 50:
            for block in range(len(depth)):
                for i in range(depth[block]):
                    if layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)
                    conv = self.bottleneck_block(
                        input=conv,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        name=conv_name,
                        data_format=data_format)

        else:
            for block in range(len(depth)):
                for i in range(depth[block]):
                    conv_name = "res" + str(block + 2) + chr(97 + i)
                    conv = self.basic_block(
                        input=conv,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        is_first=block == i == 0,
                        name=conv_name,
                        data_format=data_format)

        pool = fluid.layers.pool2d(
            input=conv,
            pool_type='avg',
            global_pooling=True,
            data_format=data_format)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        out = fluid.layers.fc(
            input=pool,
            size=class_dim,
            param_attr=fluid.param_attr.ParamAttr(
                name="fc_0.w_0",
                initializer=fluid.initializer.Uniform(-stdv, stdv)),
            bias_attr=ParamAttr(name="fc_0.b_0"))
        return out

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None,
                      name=None,
                      data_format='NCHW'):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            param_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False,
            name=name + '.conv2d.output.1',
            data_format=data_format)

        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        return fluid.layers.batch_norm(
            input=conv,
            act=act,
            name=bn_name + '.output.1',
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance',
            data_layout=data_format)

    def shortcut(self, input, ch_out, stride, is_first, name, data_format):
        if data_format == 'NCHW':
            ch_in = input.shape[1]
        else:
            ch_in = input.shape[-1]
        if ch_in != ch_out or stride != 1 or is_first == True:
            return self.conv_bn_layer(
                input, ch_out, 1, stride, name=name, data_format=data_format)
        else:
            return input

    def bottleneck_block(self, input, num_filters, stride, name, data_format):
        conv0 = self.conv_bn_layer(
            input=input,
            num_filters=num_filters,
            filter_size=1,
            act='relu',
            name=name + "_branch2a",
            data_format=data_format)
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu',
            name=name + "_branch2b",
            data_format=data_format)
        conv2 = self.conv_bn_layer(
            input=conv1,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None,
            name=name + "_branch2c",
            data_format=data_format)

        short = self.shortcut(
            input,
            num_filters * 4,
            stride,
            is_first=False,
            name=name + "_branch1",
            data_format=data_format)

        return fluid.layers.elementwise_add(
            x=short, y=conv2, act='relu', name=name + ".add.output.5")
  
    
def ResNeSt50():
    model = ResNeSt(layers=[3,4,6,3], radix=2, groups=1, bottleneck_width=64, 
                      deep_stem=True, stem_width=32, avg_down=True,
                      avd=True, avd_first=False, final_drop=0.2, dropblock_prob=0.0)
    return model


def ResNeSt101():
    model = ResNeSt(layers=[3,4,23,3], radix=2, groups=1, bottleneck_width=64,
                       deep_stem=True, stem_width=64, avg_down=True,
                       avd=True, avd_first=False, final_drop=0.2, dropblock_prob=0.0)
    return model


def ResNeSt200():
    model = ResNeSt(layers=[3,24,36,3], radix=2, groups=1, bottleneck_width=64,
                        deep_stem=True, stem_width=64, avg_down=True,
                        avd=True, avd_first=False, final_drop=0.2, dropblock_prob=0.2)
    return model

def ResNeSt269():
    model = ResNeSt(layers=[3,30,48,8], radix=2, groups=1, bottleneck_width=64,
                        deep_stem=True, stem_width=64, avg_down=True,
                        avd=True, avd_first=False, final_drop=0.2, dropblock_prob=0.2)
    return model


if __name__ == "__main__":

    image = fluid.data(name='image', shape=[16, 3, 224, 224], dtype='float32')
    
    model = ResNeSt269()
    out = model.net(input=image, class_dim=1000)
    test_program = fluid.default_main_program().clone(for_test=True)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    #lltotal_flops_params, is_quantize = summary(test_program)
    fluid.save(test_program, "test_resnest269")