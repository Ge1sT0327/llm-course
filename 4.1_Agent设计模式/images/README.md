# 课程配图

## 已有Mermaid图源文件

- `plan_solve.mmd`
- `react_loop.mmd`

## 渲染方法

### 方法1: Mermaid Live (在线)
1. 打开 [mermaid.live](https://mermaid.live)
2. 复制 `.mmd` 文件内容粘贴到编辑器
3. 导出为 PNG/SVG

### 方法2: VS Code
1. 安装插件 "Markdown Preview Mermaid Support"
2. 在 Markdown 中使用 ```` ```mermaid ```` 代码块
3. 预览并截图

### 方法3: Mermaid CLI
```bash
npm install -g @mermaid-js/mermaid-cli
mmdc -i input.mmd -o output.png
```

## 配图清单

| 图名 | 类型 | 说明 |
|------|------|------|
| Plan Solve | 架构图/流程图 | 见 plan_solve.mmd |
| React Loop | 架构图/流程图 | 见 react_loop.mmd |
