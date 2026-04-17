# 部署到阿里云 OSS + CDN

## 一、阿里云控制台操作（约 10 分钟）

### 1. 创建 OSS Bucket

1. 登录 [OSS 控制台](https://oss.console.aliyun.com/)
2. 点击「创建 Bucket」
3. 配置：
   - Bucket 名称：`cot-folding-demo`（全局唯一，改成你想要的名字）
   - 地域：选离用户近的（如 `华东2-上海`）
   - 存储类型：标准存储
   - 读写权限：**公共读**
   - 其他保持默认
4. 点击「确定」

### 2. 开启静态网站托管

1. 进入 Bucket → 基础设置 → 静态页面
2. 开启「默认首页」，设为 `index.html`
3. 「默认 404 页」也设为 `index.html`（SPA 路由兜底）
4. 保存

### 3. 获取 AccessKey

1. 进入 [RAM 控制台](https://ram.console.aliyun.com/manage/ak)
2. 创建 AccessKey（或用已有的）
3. 记下 AccessKey ID 和 AccessKey Secret

### 4.（可选）绑定自定义域名 + CDN

如果需要自定义域名（如 `demo.yourlab.com`）：

1. 进入 Bucket → 传输管理 → 域名管理
2. 绑定自定义域名
3. 开启 CDN 加速
4. 在域名服务商处添加 CNAME 记录

如果不需要自定义域名，直接用 OSS 提供的域名即可：
`https://cot-folding-demo.oss-cn-shanghai.aliyuncs.com/index.html`

---

## 二、本机操作

### 1. 安装 oss2 SDK

```bash
pip install oss2
```

### 2. 配置凭证

```bash
python deploy_oss.py --configure
```

按提示输入：
- AccessKey ID
- AccessKey Secret
- Endpoint（如 `oss-cn-shanghai.aliyuncs.com`）
- Bucket 名称（如 `cot-folding-demo`）
- CDN 域名（可选，没有就直接回车）

凭证保存在 `.oss_config.json`（已 chmod 600，不会被 git 追踪）。

### 3. 首次部署（完整上传）

```bash
# 1. 构建前端
npx vite build

# 2. 预压缩前端 + 数据文本资源（JS/CSS/HTML/JSON/sim.b64）
python deploy_oss.py --gzip

# 3. 配置 Bucket + 上传全部文件
python deploy_oss.py --setup --upload
```

或者一步到位：

```bash
npx vite build && python deploy_oss.py --gzip --upload --setup
```

预计耗时：
- gzip 压缩：~5 分钟
- 上传 ~1.3 GB：取决于带宽，30Mbps 上行约 6-8 分钟

### 4. 后续更新（只改了前端代码）

```bash
npx vite build && python deploy_oss.py --gzip --upload --frontend-only
```

只上传前端文件，且 JS/CSS/HTML 也会带 `Content-Encoding: gzip`。

---

## 三、费用参考

| 项目 | 单价 | 估算 |
|------|------|------|
| OSS 存储（1.3GB gzip） | ¥0.12/GB/月 | ¥0.16/月 |
| OSS 请求（GET） | ¥0.01/万次 | ¥0.1/月（1万PV） |
| CDN 流量 | ¥0.24/GB | 按用量 |
| CDN 100GB 流量包 | ¥16（有效1年） | 约够 1.5 万次访问 |
| 域名（可选） | ¥30-60/年 | |

低流量场景（实验室 + 审稿人）：**月费 < ¥5**

---

## 四、文件结构说明

上传到 OSS 的目录结构：

```
/                           ← Bucket 根目录
├── index.html              ← 入口（no-cache）
├── assets/                 ← JS/CSS（immutable, 1年缓存）
│   ├── index-xxx.js
│   ├── index-xxx.css
│   └── ...
└── data/aime24/            ← 静态数据（immutable, 30天缓存）
    ├── app.json
    ├── overview.json
    ├── problems.index.json
    ├── compare/p60.json ... p89.json
    └── samples/
        ├── p60/s0.bundle.json ... s63.bundle.json
        ├── p60/s0.text.json   ... s63.text.json
        └── ...
```

所有文本资源（`index.html`、`assets/*.js`、`assets/*.css`、JSON、`*.sim.b64`）都建议以 gzip 压缩格式上传（`Content-Encoding: gzip`），浏览器自动解压，用户无感知。

---

## 五、验证

部署完成后访问：

```
https://<bucket>.oss-cn-<region>.aliyuncs.com/index.html
```

或自定义域名：

```
https://demo.yourlab.com/
```

检查项：
- [ ] 首页加载正常
- [ ] 切换 Problem/Sample 能加载可视化
- [ ] 点击 slice 能显示文本
- [ ] Dark mode 正常
- [ ] Batch Overview 弹窗正常
