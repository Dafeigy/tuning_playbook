# Deep Learning Tuning Playbook

*这并非为受谷歌官方的支持的产品。*

**Varun Godbole<sup>&dagger;</sup>, George E. Dahl<sup>&dagger;</sup>, Justin Gilmer<sup>&dagger;</sup>, Christopher J. Shallue<sup>&Dagger;</sup>, Zachary Nado<sup>&dagger;</sup>**


&dagger; Google Research, Brain Team

&Dagger; Harvard University

## 文档内容

-   [Who is this document for?](#who-is-this-document-for)
-   [Why a tuning playbook?](#why-a-tuning-playbook)
-   [Guide for starting a new project](#guide-for-starting-a-new-project)
    -   [Choosing the model architecture](#choosing-a-model-architecture)
    -   [Choosing the optimizer](#choosing-the-optimizer)
    -   [Choosing the batch size](#choosing-the-batch-size)
    -   [Choosing the initial configuration](#choosing-the-initial-configuration)
-   [A scientific approach to improving model performance](#a-scientific-approach-to-improving-model-performance)
    -   [The incremental tuning strategy](#the-incremental-tuning-strategy)
    -   [Exploration vs exploitation](#exploration-vs-exploitation)
    -   [Choosing the goal for the next round of experiments](#choosing-the-goal-for-the-next-round-of-experiments)
    -   [Designing the next round of experiments](#Designing-the-next-round-of-experiments)
    -   [Determining whether to adopt a training pipeline change or
        hyperparameter
        configuration](#Determining-whether-to-adopt-a-training-pipeline-change-or-hyperparameter-configuration)
    -   [After exploration concludes](#After-exploration-concludes)
-   [Determining the number of steps for each training run](#Determining-the-number-of-steps-for-each-training-run)
    -   [Deciding how long to train when training is not compute-bound](#Deciding-how-long-to-train-when-training-is-not-compute-bound)
    -   [Deciding how long to train when training is compute-bound](#Deciding-how-long-to-train-when-training-is-compute-bound)
-   [Additional guidance for the training pipeline](#Additional-guidance-for-the-training-pipeline)
    -   [Optimizing the input pipeline](#Optimizing-the-input-pipeline)
    -   [Evaluating model performance](Evaluating-model-performance)
    -   [Saving checkpoints and retrospectively selecting the best checkpoint](#Saving-checkpoints-and-retrospectively-selecting-the-best-checkpoint)
    -   [Setting up experiment tracking](#Setting-up-experiment-tracking)
    -   [Batch normalization implementation details](#Batch-normalization-implementation-details)
    -   [Considerations for multi-host pipelines](#Considerations-for-multi-host-pipelines)
-   [FAQs](#faqs)
-   [Acknowledgments](#acknowledgments)
-   [Citing](#citing)
-   [Contributing](#contributing)

## 这篇文档的受众？

此文档是为对**最大化深度学习模型性能**感兴趣的工程师和（个人与团队）研究者提供的。我们假定您已拥有基础的机器学习和深度学习概念的相关认识。本文档的重点在于**超参数调整的过程**。文档同样涉及到深度学习训练的另一方面进行了探索，比如训练的方法部署和优化，但是这部分的内容并非我们完善的重点。我们假定机器学习问题是一个监督学习问题或一个与之类似的问题（例如自监督问题），这意味着本文档内的一些方法可能同样适用于其他类型的问题中。

## 为何是一本playbook？

目前，已有大量的工作和猜想的实践使深度神经网络在实践中良好地工作。然而，使用深度学习获得良好结果的实现方法很少被记录下来。论文为了更好地组织结构忽略了训练过程的描述而直接展现最终结果，与此同时聚焦于商业项目上的机器学习工程师很少有时间记录下他们的过程。教科书倾向于回避实践指导，而是优先考虑基础知识原则，即便这些作者有良好的应用经验并可以提供给有用的建议。在准备这篇文档时，我们无法找到一种能去全面解释 *如何在深度学习中获得良好结果* 的方法。相反，我们发现在博客帖子与社交媒体上找到了一些建议，包括在论文附录中的训练技巧、某个研究项目的特定训练流程以及诸多的困惑。深度学习专家和缺乏能力的实践者使用看似相同的方法在结构上也存在着鸿沟；同时，这些专家也承认他们所使用的一些技巧可能并不是有所依据的。随着深度学习的成熟并将对世界产生更大的影响，我们的社区需要更多涵盖所有对获得良好结果至关重要的实用细节和有效方法的资源。

我们是一个耕耘深度学习领域多年的五人研究者与工程师的团队，我们中的一些人早在2006年就开始了相关的研究。从语音识别到天文学，我们将深度学习应用于各种问题，并在这个过程中积累了许多知识与经验。这篇文档源于我们自己训练神经网络的经验，旨在指导机器学习的新工程师并为我们的同时提供深度学习的实践建议。尽管看到在大量的高校实验室里实践机器学习的方法和机器学习驱动的技术产品造福上亿人非常欣慰，深度学习作为一门工程学科依然处于起步阶段。我们希望这份文档可以鼓励并帮助他人理解该领域的实验方法。

这份文档是在我们试图凝练自己在深度学习的方法时产生的，因此它代表了作者当时的观点，而不是任何形式的客观事实。我们自己与超参数的调优斗争使它成为我们指南的一个特别重点，但此文档也涵盖其他我们在工作中遇到（或看到出错）的重要问题。我们的意图是让这项工作成为一份灵活的文档，并随着我们的观点会改变。例如，两年前我们如果要写关于调试和调优的材料来说是不可能的，因为它需要基于最近的研究结果和正在进行的调查。不可避免地，我们的一些建议将需要更新，以说明新的结果和改进工作流程。我们不知道*最佳的*深度学习方法，但是只要社区没开始记录并讨论不同的流程，我们就不能找到它。为此，我们鼓励那些使用我们的提出的建议并发现了更好的方法的读者，带着令人信服的证据，一起更新这篇文档。我们也希望看到其他的指南和提供可能不同建议的文档，这样我们就一起将实践做到最好。最后，任何标有🤖表情符号是我们想做更多的研究的方向。只有在我们尝试写下这篇文档之后才发现，在深度学习从业者的工作流程中可以找到多少有趣的和被忽视的研究问题。

## 开始新项目的指南

调优过程的许多决定都可以在项目开始之际决定，仅需在场景变化时偶尔需要重新考虑。

我们下文的内容有如下的假设：

-   问题的建模足够本质，数据清洗之类的工作已经完成，并且已经在模型架构和训练配置上花费实践验证有效；
-   已经建立了包含训练和评估的流程，并且对不同的研究模型容易执行训练和预测的工作；
-   选择并部署了恰当的测量手段，其中测量手段应该能充分反映部署环境下的需测量的信息。

### 选择模型架构

***总结：*** *当开始一个新项目时，尝试重新使用一个已经有效的模型。*

-   先选择一个常见的模型架构、建立好的模型。自定义的模型通常都可以在之后构建。
-   模型架构通常有许多超参数决定了模型的大小和其他细节（比如说层数、层宽、激活函数等）
    -   因此，选择模型架构意味着选择不同模型系列（通过超参数设定产生的同类型模型）。
    -   我们会在[Choosing the initial configuration](#choosing-the-initial-configuration)和[A scientific approach to improving model performance](#a-scientific-approach-to-improving-model-performance)两个模块讨论模型超参数的选择问题。
-   如果可以的话，找到和你需要解决问题相似的论文并将论文中的模型作为一个调试的出发点。

### 选择优化器

***总结:*** *就当前问题选择最欢迎的优化器。*

-   在不同机器学习问题和模型架构的背景下没有最“优”的优化器。即便是对比优化器的性能就是一个复杂的问题，详见论文[comparing the performance of optimizers is a difficult task](https://arxiv.org/abs/1910.05446)。🤖
-   我们建议使用构建良好的受欢迎的优化器，尤其是项目开始之时。
    -   理想情况下，对同类型问题选择最受欢迎的优化器。
-   注意所选优化器涉及的**所有**超参数：
    -   有更多超参数的优化器需要更多的调试以得到最好的配置。
    -   项目的初始阶段将优化器的超参数作为 [待调优参数](#identifying-scientific-nuisance-and-fixed-hyperparameters)并将其他超参数（比如说模型架构的超参数）的值调优非常关键。
    -   在训练开始之时选择简单的优化器（比如说固定动量的随机梯度下降或者固定$\epsilon$, $\beta_{1}$以及$\beta_{2}$的 Adam 优化器）并在之后使用更通用的优化器。
-   我们喜欢使用的一些构造良好的优化器（包括但不限于）有：
    -   [带动量的 SGD ](#what-are-the-update-rules-for-all-the-popular-optimization-algorithms)（我们喜欢使用Nesterov的变种）
    -   [Adam 和 NAdam](#what-are-the-update-rules-for-all-the-popular-optimization-algorithms)，比带动量的 SGD 更具适用性。 注意： Adam 有4个可调参数并且[他们都很有用](https://arxiv.org/abs/1910.05446)！
        -   参考：
            [How should Adam's hyperparameters be tuned?](#how-should-adams-hyperparameters-be-tuned)

### 选择批次大小

***总结：*** *批次大小决定了训练的速度并且不应该在为了提高验证集性能而直接更改。通常来说理想的批次大小将会是可行硬件支持的最大批次大小。*

-   批次大小是一个对 *训练时间* 长短和 *计算资源消耗* 的关键决定因素。
-   增加批次大小通常会减少训练时间。这对训练而言很有用，比如说：
    -   允许超参数更彻底地在一个固定时间间隔内调整，并能隐式地训练出更好的模型。
    -   减少开发过程的延迟，新的想法可以更快地得到测试验证。
-   增加批次大小导致的资源消耗可能会或减少、或增加或甚至不变。
-   批次大小 *不应该* 被当作验证集性能上的一个可调的超参数。
    -   只要超参数被调整好（尤其是学习率和正则参数）并且训练的步骤是合理的，使用任意不同的批次大小应该会获得相同的效果（见[Shallue 等人的论文](https://arxiv.org/abs/1811.03600)）。
    -   请参考 [Why shouldn't the batch size be tuned to directly improve validation set performance?](#why-shouldnt-the-batch-size-be-tuned-to-directly-improve-validation-set-performance)这一节内容。

#### 确定可行的批次大小并估量训练的吞吐

-   对于给定的模型和优化器，通常由很多的硬件支持的批次大小选择。批次大小的限制因素主要是加速器的内存。（译注：一般指GPU的显存）
-   不幸的是，不运行代码或者至少编译整个训练程序，都无法计算合适的批次大小。
-   最简单的方法就是通常在运行训练任务时设定不同的批次大小（比如按2的幂次方选取）并在小范围内运行指导其中一项任务达到了可行的加速器内存上限。
-   对于每一个批次大小，我们应该训练足够长的时间以获得准确的 *训练吞吐* （training throughput）的估计：

<p align="center"><em>训练吞吐</em> = 每秒处理的样本数</p>

<p align="center">或者等价地计算 <em>每步消耗的时间（time per step）</em>：</p>

<p align="center"><em>每步消耗的时间</em> = (批次大小) / (训练吞吐)</p>

-   当加速器还未满载时，如果批次大小加倍那么训练的吞吐量应该也是（或者至少接近）翻倍的。等价的，每步执行的时间随着批次大小增加，应该增加（或者至少接近）一个常数。
-   如果不符合上述描述情况，那么训练的流程存在某种瓶颈，比如说系统的I/O问题或者不同计算节点的同步问题，这需要在真正训练之前诊断并纠正回来。
-   如果训练吞吐的增加和某些最大的批次大小相关，那么我们应该只考虑最大批次大小相关的批次大小选项，即便我们的硬件支持更大的批次大小。
    -   所有使用更大的批次大小带来的好处都是假定在训练吞吐增加的基础上的。如果训练吞吐增加并不会带来更好的效果，那么应该去修复瓶颈或者使用更小的批次大小。
    -   **梯度积累** 模拟了一个比硬件可支持的更大的批次大小，因此它并不能提供任何吞吐上的增益。它在应用过程中应该避免被使用。
-   这些步骤需要在模型或者优化器产生变化时进行重复（比如说一个不同的模型架构可以允许内存容纳更大的批次大小）。

#### 选择最小化训练时间的批次大小

<p align="center">训练时间 = (每步执行时间) x (执行步骤总数)</p>


-   我们通常对不同的可行批次大小将每步执行时间粗略认为是一个常数，这种假设是合理的因为并行计算没有额外开销，并且训练瓶颈已经被攻克(如何确定训练的瓶颈请参见[前一节内容](#determining-the-feasible-batch-sizes-and-estimating-training-throughput)）。在实践中通常至少会有一些随批次大小增加而产生的开销。
-   当批次大小增加时，达到相同性能所需要的训练总步数会下降（当批次大小改变时所有相关的超参数被重新调整的情况，参考[Shallue 等人的论文](https://arxiv.org/abs/1811.03600)）。
    -   比如说，批次大小翻倍可能会使训练总步数减半，这称为**完美尺度变换**。
    -   完美尺度变换有一个批次大小的阈值，超出这个阈值 将不会有训练步数的增益。
    -   最后，无限地增加批次大小不会减少训练的总步数（但至少不会增加）。
-   因此最小化训练时间的批次大小通常就是能使得训练步数递减最多的最大批次大小。
    -   这个批次大小取决于数据集、模型和优化器，并且如何计算这个批次大小是一个开放的问题而不是在面对新问题时通过实验得到的。🤖
    -   当比较批次大小时，要注意样本/[回合](https://developers.google.com/machine-learning/glossary#epoch)开销和单步执行开销的区别。前者是在固定数量的训练样本上进行的所有实验，后者是在训练步数固定时运行进行的所有实验。
        -   即便有更大的批次大小能提供减少训练步骤的所需要的时间的方法，对比批次大小的回合开销也只和完美尺度缩放有关。
    -   通常来说，最大的硬件支持的批次大小会比上文提到的阈值要小。因此一个绝好的方法（不需要额外的实验来验证）就是用尽可能大的批次大小。
-   某些更大的批次大小却导致训练总时间增加，这种情况一定要避免使用。

#### 选择最小化资源开销的批次大小

-   和批次大小增加有关的资源开销有两种：
    1.  先期开销，比如使用新的硬件或在多GPU/多TPU上训练以覆写训练流程。
    2.  后期开销，比如团队资源分配、云服务器、电力/维护开销。
-   如果先期开销会随着批次大小的增加而增加，那么最好在项目成熟前拖延批次大小的增加，这更易于在成本-收益之间获得权衡。部署多主机的并行训练程序可能会引入[bugs](#considerations-for-multi-host-pipelines) 或一些微小的[issues](#batch-normalization-implementation-details) 并且最好一开始时使用简单的训练流程。（从另一方面来讲，训练时间的加速可能在早期非常有用因为此时需要大量的调参实验。）
-   我们将后期开销 （包含多种开销）称为资源消耗。 资源消耗的计算可以依据下式进行：

<p align="center">资源消耗 = (每步资源消耗) x (总执行步数)</p>

-   增加批次大小通常会 [减少训练步骤总数](#choosing-the-batch-size-to-minimize-training-time)。不管资源消耗是增加还是减少，都取决于每一步的资源消耗的变化。
    -   Increasing the batch size might *decrease* the resource consumption. For
        example, if each step with the larger batch size can be run on the same
        hardware as the smaller batch size (with only a small increase in time
        per step), then any increase in the resource consumption per step might
        be outweighed by the decrease in the number of steps.
    -   增加批次大小并 *不能改变* 资源消耗。比如说，如果将批次大小加倍会将训练的步骤数减半并增加了GPU的使用数量，GPU使用时间的总消耗并不会改变。
    -   增加批次大小可能会 *增加* 资源消耗，比如说，如果增加批次大小需要以升级硬件设施作为代价，那么每步资源消耗的增加可能会超过总的训练步数。

#### 改变批次大小需要重新调整大部分的超参数


-   大多数的超参数的最优质对批次大小都很敏感。因此，改变批次大小可能需要重新开始调参过程。
-   与批次大小最为相关的超参数，即需要重点对不同批次大小进行调整的超参数，就是和优化其相关的超参数（比如说学习率，动量等）以及正则化参数。
-   一定要记住在项目开始之时当选择批次大小时，如果需要在后期调整批次大小使用新的批次大小，这可能会造成额外的时间消耗和重新调参的资源消耗。

#### 批次归一化如何与批次大小相关


-   批次归一化很复杂，并且通常时使用不同的批次大小而不是梯度计算去计算统计量（compute statistics）。详情请参考[batch norm section](#batch-normalization-implementation-details) 部分的内容。

### 选择初始配置

-   在开始调整潮参数之前必须先确定出发点。包括(1)模型的配置（比如说模型的层数），(2)优化器的参数（比如说学习率），(3)训练步数。
-   确定初始配置需要人工参与到训练配置和试错过程。
-   我们的指导目的是帮助找到一种简单、相对快速、较少资源消耗并能得到一个合理的结果的的配置方法。
    -   “简单”意味着避免任何额外的功能的使用，因为这些可以在后期添加上。即便额外的功能能有效帮助提升模型效果，在训练初期使用它们其实有浪费时间调参或引入不必要的计算复杂度的风险和问题。
        -   比如说，使用一个固定的学习率衰减常数进行训练，而不是使用各种花里胡哨的学习率迭代方法更改学习率。
    -   选择快速且资源消耗少的初始配置，这能帮助调参更加高效进行。
        -   比如说，使用一个更小的模型开始训练。
    -   合理的”性能表现取决于问题，但至少要保证训练好的模型在验证集上的表型要比随机选择的结果好（尽管效果很差并不值得部署）。
-   训练的步数选择需要考虑如下因素的平衡：
    -   从另一个角度说，更多步骤的训练能提高性能并且能使超参数调整更简单 (见 [Shallue 等人的论文](https://arxiv.org/abs/1811.03600))。
    -   另一方面，更少步骤的训练意味着更少的资源消耗和更快的训练速度，通过减少训练间隔时间加速调参性能并允许并行地进行实验。而且如果在初始选择了不必要的大额开销（step budget），那么很难在训练过程中更改，比如说一开始选择了和训练步骤相关的学习率迭代方法。

## 一种提高模型性能的科学方法

机器学习的发展的终极目标就是最大化发挥部署模型的功能。即便和其他应用的开发（比如说开发时间、可行的计算资源、模型类型）有很多区别，我们同样可以针对问题使用一些相同基础的方法以及规则。

我们的指南依照如下假设进行：

-   已经存在一个完全可行的训练方法和可获得合理结果的配置。
-   有足够的计算资源以完成调参实验并且能并行地运行多个训练任务。

### 增量调参策略

***总结:*** *开始时使用一个简单的配置并在针对问题逐渐增加额外的提升因素。确保任何的提升都有可靠的证据支撑，并且避免引入不必要的复杂度。*

-   我们的最终目的是找到一个能最大化发挥模型性能的配置。
    -   在某些情况下，我们的目标是在期限时间内最大化模型性能。
    -   在另一些情况下，我们希望模型能够持续地不断提升性能（比如说不断地在使用过程中提升性能）。
-   原则上来说，我们可以使用一种算法来自动搜索整个可行的配置空间中的配置，但是这不是一个实践中会考虑的方法。
    -   可行的配置空间非常大并且暂时还没有一种算法能够在没有人类干预情况下精确地高效搜索到这个配置。
-   大多数的自动搜索算法依赖人类设计的 *搜索空间* ，这个人为规定的配置空间对决定搜索算法搜索结果的性能有重要影响。
-   最有效的最大化模型性能方法就是使用一个简单的配置开始训练，并在训练过程中逐渐增加功能和提升方法。
    -   我们使用自动搜索方法在每一轮调参并不断更新搜索空间，并把更新结果作为我们的理解增长的表现。
-   随着我们的探索我们很自然地找到越来越好的配置，然后我们最“好”的模型性能也不断地提高。
    -   我们称我们更新时得到的最好配置为一个 *发行(launch)* ，这个最好的配置可能或也可能不是生产过程中的实际的发行。
    -   For each launch, we must make sure that the change is based on strong
        evidence – not just random chance based on a lucky configuration – so
        that we don't add unnecessary complexity to the training pipeline.对于每一个发行，我们必须确保每一个改变都有可靠的证据支撑-不是运气好捧出来的一个随机配置-因此我们不会在训练流程中添加额外的不必要的复杂度。

从高级的角度来讲，我们的增量调参策略就是如下四个步骤的重复：

1.  确定下一轮实验的目标的合适范围。
2.  设计并且运行一系列的实验使得模型朝目标方向移动。
3.  从实验结果中分析原因。
4.  考虑是否发行最好的配置。

本节的后续内容将更详细地描述这种策略。

### 探索 vs 验证

***总结:*** *大多数时候我们的主要目标是深入地了解一个问题。*

-   尽管有人会认为我们使用了太多时间来最大化验证集上的性能，但我们在实践中我们尝试花更多的时间深入了解问题，并相对花费较少的精力解决验证集错误的问题。
    -   从另一方面来讲，我们大多数的时间使用在“探索”，而“验证”只占了很少一部分。
-   长期来看，如果要最大化模型最终i性能，理解问题更重要。短期内对问题的深入了解有助于我们：
    -   避免性能良好的过程中仅通过训练过程中偶尔的意外引入不必要的改变。
    -   确定验证集错误中对超参数最敏感的因素，哪些超参数需要以及和其他超参数一起被重新调整，哪些超参数不那么敏感可以在后面的实验中设定为常量。
    -   尝试一些可能有效的功能，比如说如果出现了过拟合尝试使用新的正则化方法。
    -   确认对结果影响无关的特征并将其移除，为后来的实验减少复杂度。
    -   辨识出超参数调整对模型性能提升的饱和情况。
    -   缩小搜索空间到最优值附近以提高调参效率。
-   当我们最终决定“贪婪”尝试新方法，我们可以将注意力完全放在验证集错误，即便实验还没有针对调参问题的本质得到最有用的信息。

### 选择下一轮实验的目标

***总结:*** *每一轮实验都应该有一个清晰的目标并且可以通过实验逼近这个目标并有效缩小问题的规模。*

-   每一轮实验都应该有一个清晰的目标并且可以通过实验逼近这个目标并有效缩小问题的规模：如果我们尝试同时增加更多的特征或回答更多的问题，我们可能不能分辨出究竟是哪一个在发挥作用。
-   目标的案例包括：
    -   尝试在训练流程使用通用的提升方法（比如说新的正则项，预处理选项等等）
    -   理解某一个模型的超参数的影响（比如说激活函数）
    -   贪婪地最大化验证集误差。

### 设计下一轮实验

***总结:*** *确定哪些超参数是研究对象（scientific）、无关干扰（nuisance）以及固定常量（fuxed）。在忽略无关的超参时优化创建一个研究的序列来对比不同研究对象超参数取值的影响，通过给定的研究对象超参选择无关超参的搜索空间以平衡资源消耗。*