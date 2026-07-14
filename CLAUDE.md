# CLAUDE.md

## 核心规则（每次必须遵守）

### 写代码
- 先 Brainstorm 理解需求 → Plan 出计划经用户确认 → TDD 先写测试 → 实现 → Verify 跑验证 → Code Review
- **不测试，不提交**

### 行为约束
- 用户要 A 就做 A，有更好方案先问再改，不擅自替换
- 发现用户做法有风险或低效时主动反问
- 所有回复用中文
- 文件默认放 D 盘

### Agent 协作（调用任何外部 Agent/模型时）
- 必须先展示完整上下文给用户确认（项目背景→当前状态→任务→上下文→约束→验收，六段缺一不可）
- 我自己必须并行分析，不等不靠
- Agent 输出必须和我独立分析交叉比对，不原样转发

### 后台任务
- 命令切后台后主动跟进，跑完汇报结果，不能当没发生

## 项目约定
- Python 3.13 + Flask + OpenCV + SAM (Segment Anything Model)
- 项目目的：婴儿头型分析，通过照片提取头部轮廓并与标准头型对比
- 标准头型图是红色虚线示意图，不是真实照片——不要对它跑 SAM
- 真实测试照片：real_photo_2.jpg 等
- 关键文件：app.py（主入口）、sam_detector.py（SAM 分割）、standard_compare.py（标准对比）、head_analyzer.py（分析）
