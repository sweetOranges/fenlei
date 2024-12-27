#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <getopt.h>
// 定义网络结构

// 文件路径定义
char *MODEL_FILE;
int INPUT_SIZE = 0;
int HIDDEN_SIZE = 0;
int OUTPUT_SIZE = 0;
double LEARNING_RATE = 0.01;
int EPOCHS = 100000;
int MAX_SAMPLES = 2800;  // 最大样本数量
int LINE_BUFFER = 10240; //// 每行最大字符数

// 神经网络结构体
typedef struct
{
    int input_size;
    int hidden_size;
    int output_size;
    double **hidden_weights;
    double *hidden_bias;
    double **output_weights;
    double *output_bias;
} NeuralNetwork;

// 激活函数及其导数
double relu(double x)
{
    return x > 0 ? x : 0;
}

double relu_derivative(double x)
{
    return x > 0 ? 1.0 : 0.0;
}

// Softmax函数
void softmax_func(double *input, double *output, int length)
{
    double max = input[0];
    for (int i = 1; i < length; i++)
    {
        if (input[i] > max)
            max = input[i];
    }
    double sum = 0.0;
    for (int i = 0; i < length; i++)
    {
        output[i] = exp(input[i] - max); // 稳定性处理
        sum += output[i];
    }
    for (int i = 0; i < length; i++)
    {
        output[i] /= sum;
    }
}

// 随机初始化权重
double random_weight()
{
    return ((double)rand() / RAND_MAX) * 2 - 1; // [-1, 1]
}


void init_model(NeuralNetwork *nn)
{
    nn->input_size = INPUT_SIZE;
    nn->hidden_size = HIDDEN_SIZE;
    nn->output_size = OUTPUT_SIZE;
    nn->hidden_weights = (double **)malloc(nn->input_size * sizeof(double *));
    for (int i =0 ;i < nn->input_size; i++)
    {
        nn->hidden_weights[i] = (double *)malloc( nn->hidden_size * sizeof(double));
    }
    nn->hidden_bias = (double *)malloc(nn->hidden_size * sizeof(double));
    nn->output_weights = (double **)malloc(nn->hidden_size  * sizeof(double *));
    for (int i =0 ;i < nn->hidden_size; i++)
    {
        nn->output_weights[i] = (double *)malloc( nn->output_size * sizeof(double));
    }
    nn->output_bias = (double *)malloc(nn->output_size * sizeof(double));
}

// 保存模型到文件
int save_model(NeuralNetwork *nn, const char *filename)
{
    FILE *fp = fopen(filename, "wb");
    if (fp == NULL)
    {
        printf("无法打开文件 %s 进行写入。\n", filename);
        return -1;
    }
    fwrite(&nn->input_size, sizeof(int), 1, fp);
    fwrite(&nn->hidden_size, sizeof(int), 1, fp);
    fwrite(&nn->output_size, sizeof(int), 1, fp);
    for (int i = 0; i < nn->input_size; i++)
    {
        fwrite(nn->hidden_weights[i], sizeof(double), nn->hidden_size, fp);
    }
    fwrite(nn->hidden_bias, sizeof(double), nn->hidden_size, fp);
    for (int i = 0; i < nn->hidden_size; i++)
    {
        fwrite(nn->output_weights[i], sizeof(double), nn->output_size, fp);
    }
    fwrite(nn->output_bias, sizeof(double), nn->output_size, fp);
    fclose(fp);
    return 0;
}

// 从文件加载模型
int load_model(NeuralNetwork *nn, const char *filename)
{
    FILE *file = fopen(filename, "rb");
    if (file == NULL)
    {
        printf("无法打开文件 %s 进行读取。\n", filename);
        return -1;
    }
    // Read the sizes of the layers
    fread(&INPUT_SIZE, sizeof(int), 1, file);
    fread(&HIDDEN_SIZE, sizeof(int), 1, file);
    fread(&OUTPUT_SIZE, sizeof(int), 1, file);
    init_model(nn);

    for (int i = 0; i < nn->input_size; i++)
    {
        fread(nn->hidden_weights[i], sizeof(double), nn->hidden_size, file);
    }

    fread(nn->hidden_bias, sizeof(double), nn->hidden_size, file);
    for (int i = 0; i < nn->hidden_size; i++)
    {
        fread(nn->output_weights[i], sizeof(double), nn->output_size, file);
    }
    fread(nn->output_bias, sizeof(double), nn->output_size, file);
    fclose(file);
    return 0;
}

// 加载CSV数据
int load_csv(const char *filename, double ***inputs, double ***outputs, int *sample_count)
{
    FILE *fp = fopen(filename, "r");
    if (fp == NULL)
    {
        printf("无法打开CSV文件 %s。\n", filename);
        return -1;
    }

    char line[LINE_BUFFER];
    int count = 0;

    // 首先，计算样本数量
    while (fgets(line, sizeof(line), fp))
    {
        if (strlen(line) > 1)
        { // 排除空行
            count++;
            if (count >= MAX_SAMPLES)
            {
                printf("样本数量超过最大限制 (%d)。\n", MAX_SAMPLES);
                break;
            }
        }
    }

    *sample_count = count;

    // 分配内存
    *inputs = (double **)malloc(count * sizeof(double *));
    *outputs = (double **)malloc(count * sizeof(double *));
    if (*inputs == NULL || *outputs == NULL)
    {
        printf("内存分配失败。\n");
        fclose(fp);
        return -1;
    }
    for (int i = 0; i < count; i++)
    {
        (*inputs)[i] = (double *)malloc(INPUT_SIZE * sizeof(double));
        (*outputs)[i] = (double *)calloc(OUTPUT_SIZE, sizeof(double)); // 初始化为0
        if ((*inputs)[i] == NULL || (*outputs)[i] == NULL)
        {
            printf("内存分配失败。\n");
            fclose(fp);
            return -1;
        }
    }

    // 重置文件指针到开头
    rewind(fp);

    // 读取数据
    int idx = 0;
    while (fgets(line, sizeof(line), fp) && idx < count)
    {
        if (strlen(line) <= 1)
            continue; // 排除空行

        // 去除换行符
        line[strcspn(line, "\r\n")] = 0;

        char *token = strtok(line, ",");
        int feature_idx = 0;
        int label = -1;

        while (token != NULL)
        {
            if (feature_idx < INPUT_SIZE)
            {
                (*inputs)[idx][feature_idx] = atof(token);
            }
            else if (feature_idx == INPUT_SIZE)
            {
                label = atoi(token);
                if (label < 0 || label >= OUTPUT_SIZE)
                {
                    printf("无效的标签值 %d 在样本 %d 中。\n", label, idx + 1);
                    fclose(fp);
                    return -1;
                }
                (*outputs)[idx][label] = 1.0; // One-hot编码
            }
            else
            {
                // 超出预期的字段
            }
            feature_idx++;
            token = strtok(NULL, ",");
        }

        if (feature_idx != INPUT_SIZE + 1)
        {
            printf("样本 %d 的字段数量不正确。\n", idx + 1);
            fclose(fp);
            return -1;
        }

        idx++;
    }

    fclose(fp);
    return 0;
}

// 训练函数
void train(NeuralNetwork *nn, double **training_inputs, double **training_outputs, int sample_count)
{
    // 初始化权重和偏置
    for (int i = 0; i < INPUT_SIZE; i++)
    {
        for (int j = 0; j < HIDDEN_SIZE; j++)
        {
            nn->hidden_weights[i][j] = random_weight();
        }
    }
    for (int j = 0; j < HIDDEN_SIZE; j++)
    {
        nn->hidden_bias[j] = random_weight();
    }
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        for (int j = 0; j < OUTPUT_SIZE; j++)
        {
            nn->output_weights[i][j] = random_weight();
        }
    }
    for (int j = 0; j < OUTPUT_SIZE; j++)
    {
        nn->output_bias[j] = random_weight();
    }

    // 训练网络
    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        for (int sample = 0; sample < sample_count; sample++)
        {
            // 前向传播
            double hidden_layer[HIDDEN_SIZE];
            for (int j = 0; j < HIDDEN_SIZE; j++)
            {
                hidden_layer[j] = nn->hidden_bias[j];
                for (int i = 0; i < INPUT_SIZE; i++)
                {
                    hidden_layer[j] += training_inputs[sample][i] * nn->hidden_weights[i][j];
                }
                hidden_layer[j] = relu(hidden_layer[j]);
            }

            double output_layer_raw[OUTPUT_SIZE];
            double output_layer[OUTPUT_SIZE];
            for (int j = 0; j < OUTPUT_SIZE; j++)
            {
                output_layer_raw[j] = nn->output_bias[j];
                for (int i = 0; i < HIDDEN_SIZE; i++)
                {
                    output_layer_raw[j] += hidden_layer[i] * nn->output_weights[i][j];
                }
            }
            softmax_func(output_layer_raw, output_layer, OUTPUT_SIZE);

            // 计算损失（交叉熵）并反向传播
            double output_errors[OUTPUT_SIZE];
            for (int j = 0; j < OUTPUT_SIZE; j++)
            {
                output_errors[j] = output_layer[j] - training_outputs[sample][j];
            }

            // 输出层梯度（Softmax + Cross-Entropy 的梯度简化）
            double output_deltas[OUTPUT_SIZE];
            for (int j = 0; j < OUTPUT_SIZE; j++)
            {
                output_deltas[j] = output_errors[j];
            }

            // 计算隐藏层误差
            double hidden_errors[HIDDEN_SIZE];
            for (int j = 0; j < HIDDEN_SIZE; j++)
            {
                hidden_errors[j] = 0.0;
                for (int k = 0; k < OUTPUT_SIZE; k++)
                {
                    hidden_errors[j] += output_deltas[k] * nn->output_weights[j][k];
                }
                // ReLU导数
                hidden_errors[j] *= relu_derivative(hidden_layer[j]);
            }

            // 更新输出权重和偏置
            for (int j = 0; j < HIDDEN_SIZE; j++)
            {
                for (int k = 0; k < OUTPUT_SIZE; k++)
                {
                    nn->output_weights[j][k] -= LEARNING_RATE * output_deltas[k] * hidden_layer[j];
                }
            }
            for (int j = 0; j < OUTPUT_SIZE; j++)
            {
                nn->output_bias[j] -= LEARNING_RATE * output_deltas[j];
            }

            // 更新隐藏权重和偏置
            for (int j = 0; j < INPUT_SIZE; j++)
            {
                for (int k = 0; k < HIDDEN_SIZE; k++)
                {
                    nn->hidden_weights[j][k] -= LEARNING_RATE * hidden_errors[k] * training_inputs[sample][j];
                }
            }
            for (int j = 0; j < HIDDEN_SIZE; j++)
            {
                nn->hidden_bias[j] -= LEARNING_RATE * hidden_errors[j];
            }
        }

        // 可选：每1000个epoch打印一次损失
        if (1 + 1 == 2)
        {
            double total_loss = 0.0;
            for (int sample = 0; sample < sample_count; sample++)
            {
                // 前向传播
                double hidden_layer[HIDDEN_SIZE];
                for (int j = 0; j < HIDDEN_SIZE; j++)
                {
                    hidden_layer[j] = nn->hidden_bias[j];
                    for (int i = 0; i < INPUT_SIZE; i++)
                    {
                        hidden_layer[j] += training_inputs[sample][i] * nn->hidden_weights[i][j];
                    }
                    hidden_layer[j] = relu(hidden_layer[j]);
                }

                double output_layer_raw[OUTPUT_SIZE];
                double output_layer[OUTPUT_SIZE];
                for (int j = 0; j < OUTPUT_SIZE; j++)
                {
                    output_layer_raw[j] = nn->output_bias[j];
                    for (int i = 0; i < HIDDEN_SIZE; i++)
                    {
                        output_layer_raw[j] += hidden_layer[i] * nn->output_weights[i][j];
                    }
                }
                softmax_func(output_layer_raw, output_layer, OUTPUT_SIZE);

                // 计算交叉熵损失
                for (int j = 0; j < OUTPUT_SIZE; j++)
                {
                    if (training_outputs[sample][j] == 1.0)
                        total_loss -= log(output_layer[j] + 1e-15); // 防止log(0)
                }
            }
            printf("Epoch %d, Loss: %.4f\n", epoch + 1, total_loss);
        }
    }
}

// 预测函数
int predict(NeuralNetwork *nn, double input[INPUT_SIZE])
{
    // 前向传播
    double hidden_layer[HIDDEN_SIZE];
    for (int j = 0; j < HIDDEN_SIZE; j++)
    {
        hidden_layer[j] = nn->hidden_bias[j];
        for (int i = 0; i < INPUT_SIZE; i++)
        {
            hidden_layer[j] += input[i] * nn->hidden_weights[i][j];
        }
        hidden_layer[j] = relu(hidden_layer[j]);
    }

    double output_layer_raw[OUTPUT_SIZE];
    double output_layer[OUTPUT_SIZE];
    for (int j = 0; j < OUTPUT_SIZE; j++)
    {
        output_layer_raw[j] = nn->output_bias[j];
        for (int i = 0; i < HIDDEN_SIZE; i++)
        {
            output_layer_raw[j] += hidden_layer[i] * nn->output_weights[i][j];
        }
    }
    softmax_func(output_layer_raw, output_layer, OUTPUT_SIZE);

    // 找到最大概率的类别
    int predicted = 0;
    for (int j = 1; j < OUTPUT_SIZE; j++)
    {
        if (output_layer[j] > output_layer[predicted])
            predicted = j;
    }

    return predicted;
}

extern char *optarg;
int TRAIN_FLAG = 0;
char *CSV_FILE;

void dump()
{
    printf("==============start================\n");
    printf("MODEL=%s\n", TRAIN_FLAG == 1? "train" : "test");
    printf("INPUT_SIZE=%d\n", INPUT_SIZE);
    printf("HIDDEN_SIZE=%d\n", HIDDEN_SIZE);
    printf("OUTPUT_SIZE=%d\n", OUTPUT_SIZE);
    printf("LEARNING_RATE=%f\n", LEARNING_RATE);
    printf("EPOCHS=%d\n", EPOCHS);
    printf("CSV_FILE=%s\n", CSV_FILE);
    printf("MODEL_FILE=%s\n", MODEL_FILE);
    printf("==============END================\n");
}
int main(int argc, char * const *argv)
{
    srand(time(0));
    setvbuf(stdout, NULL, _IONBF, 0);
    int c;
    int help = 0;
    char csv_file[1024] = {0};
    char model_file[1024] = {0};
    MODEL_FILE = "model.dat";
    while ((c = getopt(argc, argv, "i:h:o:l:e:m:tf:hs:")) != -1)
    {
        switch (c)
        {
        case 'i':
            INPUT_SIZE = atoi(optarg);
            break;
        case 'h':
            HIDDEN_SIZE = atoi(optarg);
            break;
        case 'o':
            OUTPUT_SIZE = atoi(optarg);
            break;
        case 'l':
            LEARNING_RATE = atof(optarg);
            break;
        case 'e':
            EPOCHS = atoi(optarg);
            break;
        case 'm':
            MAX_SAMPLES = atoi(optarg);
            break;
        case '?':
            help = 1;
            break;
        case 't':
            TRAIN_FLAG = 1;
            break;
        case 'f':
            memcpy(csv_file, optarg, strlen(optarg));
            CSV_FILE = csv_file;
            break;
        case 's':
            memcpy(model_file, optarg, strlen(optarg));
            MODEL_FILE = model_file;
            break;
        default:
            break;
        }
    }
    if (help == 1)
    {
        printf("-i INPUT_SIZE\n");
        printf("-h HIDDEN_SIZE\n");
        printf("-o OUTPUT_SIZE\n");
        printf("-l LEARNING_RATE\n");
        printf("-e EPOCHS\n");
        printf("-m MAX_SAMPLES\n");
        printf("-t train model\n");
        printf("-f train file path\n");
        printf("-s train model path\n");
        return 1;
    }
    
    double **training_inputs = NULL;
    double **training_outputs = NULL;
    int sample_count = 0;

   
    // 初始化神经网络结构体
    NeuralNetwork nn;

    if (TRAIN_FLAG == 1)
    {
        init_model(&nn);
        dump();
        // 加载CSV数据
        if (load_csv(csv_file, &training_inputs, &training_outputs, &sample_count) != 0)
        {
            printf("加载CSV数据失败。\n");
            return -1;
        }

        printf("加载了 %d 个样本\n", sample_count);
        // 训练神经网络
        train(&nn, training_inputs, training_outputs, sample_count);
        // 保存模型
        if (save_model(&nn, MODEL_FILE) == 0)
        {
            printf("模型已保存到 %s\n", MODEL_FILE);
        }
        else
        {
            printf("模型保存失败。\n");
        }
    }
    else
    {
        if (load_model(&nn, MODEL_FILE) == 0)
        {
            printf("模型加载成功。\n");
        }
        dump();
        // 加载CSV数据
        if (load_csv(csv_file, &training_inputs, &training_outputs, &sample_count) != 0)
        {
            printf("加载CSV数据失败。\n");
            return -1;
        }

        printf("加载了 %d 个样本\n", sample_count);
        // 测试网络
        printf("\n测试网络:\n");
        for (int sample = 0; sample < sample_count; sample++)
        {
            int predicted = predict(&nn, training_inputs[sample]);

            // 找到真实类别
            int actual = -1;
            for (int j = 0; j < OUTPUT_SIZE; j++)
            {
                if (training_outputs[sample][j] == 1.0)
                {
                    actual = j;
                    break;
                }
            }

            printf("输入: [");
            for (int i = 0; i < INPUT_SIZE; i++)
            {
                printf("%.2f", training_inputs[sample][i]);
                if (i < INPUT_SIZE - 1)
                    printf(", ");
            }
            printf("] 预测: %d, 真实: %d\n", predicted, actual);
        }
    }
    // 释放内存
    for (int i = 0; i < sample_count; i++)
    {
        free(training_inputs[i]);
        free(training_outputs[i]);
    }
    free(training_inputs);
    free(training_outputs);

    return 0;
}
