#include <iostream>
#include <torch/script.h>

using namespace std;

void load_script() {
    //实例化module
    torch::jit::script::Module module;
    //载入torchscript libtorch版本和pytorch版本要一致
    module = torch::jit::load(R"(/home/ubuntu/CLionProjects/libtorch-test/resnet18.torchscript)");

    //创建数据vector
    vector<torch::jit::IValue> x;
    x.emplace_back(torch::randn({ 1, 3, 224, 224 }));

    //预测
    torch::Tensor y = module.forward(x).toTensor();

    // 打印形状
    y.print();  //CPUFloatType [1, 1000]]
    //cout << y << endl;
}

int main()
{
    load_script();
    return 0;
}