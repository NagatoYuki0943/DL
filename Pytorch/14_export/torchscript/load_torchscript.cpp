/**
 * https://blog.csdn.net/zzz_zzz12138/article/details/109138805
 * 头文件:
 *		D:\ai\libtorch\include;
 *		D:\ai\libtorch\include\torch\csrc\api\include;
 * lib:
 *		D:\ai\libtorch\lib;
 *	附加依赖项:
 *		c10.lib
 *		libprotobufd.lib
 *		mkldnn.lib
 *		torch.lib
 *		torch_cpu.lib
 */


#include <iostream>
#include <torch/script.h>


using namespace std;


void load_script() {
	//实例化module
	torch::jit::script::Module module;

	//载入torchscript libtorch版本和pytorch版本要一致
	module = torch::jit::load("D:\\ai\\code\\test\\resnet18.torchscript");

	//创建数据vector
	vector<torch::jit::IValue> x;
	x.push_back(torch::randn({ 1, 3, 224, 224 }));

	//预测
	torch::Tensor y = module.forward(x).toTensor();

	cout << y << endl;
}


int main()
{
	load_script();
	system("pause");

	return 0;
}