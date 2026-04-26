#!/usr/bin/env python3
import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def default_repo_root():
    return Path(__file__).resolve().parents[1]


def default_journal_dir():
    return default_repo_root() / 'experiments' / 'research_journal'


def run_git(repo_root, args):
    result = subprocess.run(
        ['git', *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return ''
    return result.stdout.strip()


def git_snapshot(repo_root):
    return {
        'head': run_git(repo_root, ['rev-parse', 'HEAD']),
        'branch': run_git(repo_root, ['rev-parse', '--abbrev-ref', 'HEAD']),
        'status': run_git(repo_root, ['status', '--short']),
        'diff_stat': run_git(repo_root, ['diff', '--stat']),
    }


def slugify(text):
    chars = []
    for ch in text.lower():
        if ch.isalnum():
            chars.append(ch)
        elif ch in (' ', '-', '_'):
            chars.append('-')
    slug = ''.join(chars).strip('-')
    while '--' in slug:
        slug = slug.replace('--', '-')
    return slug or 'entry'


def contains_cjk(text):
    return bool(text and re.search(r'[\u4e00-\u9fff]', text))


def localize_text(text):
    if not text or contains_cjk(text):
        return text

    exact_map = {
        'PE-Anchor3DLane++ scaffold': 'PE-Anchor3DLane++ 初始脚手架',
        'Baseline reproduction and environment audit': 'Baseline 复现与环境盘点',
        'Anchor3DLane++ baseline cleanup and runbook': 'Anchor3DLane++ baseline 清理与复现说明',
        'Official baseline runner script': '官方 baseline 批量运行脚本',
        'training script yapf fallback': '训练脚本 yapf 兼容补丁',
        'distributed launcher arg compatibility': '分布式启动参数兼容修复',
        'anchor3dlanepp-r18-official-train-launch': 'Anchor3DLane++ R18 官方训练启动',
        'anchor3dlanepp-r18-official-train-relaunch': 'Anchor3DLane++ R18 官方训练重新启动',
        'anchor3dlanepp-r18-official-eval-launch': 'Anchor3DLane++ R18 官方评测启动',
    }
    if text in exact_map:
        return exact_map[text]

    prefix_map = (
        ('Evaluate official baseline ', '评测官方 baseline '),
        ('Evaluate official ', '评测官方 '),
        ('Launched official ', '启动官方 '),
        ('Relaunched official ', '重新启动官方 '),
        ('Add ', '新增'),
        ('Added ', '新增'),
        ('Audit ', '盘点'),
        ('Remove ', '清理'),
        ('Updated ', '更新'),
    )
    for prefix, replacement in prefix_map:
        if text.startswith(prefix):
            return replacement + text[len(prefix):]
    return text


def parse_metrics(metrics):
    parsed = {}
    for item in metrics:
        if '=' not in item:
            raise ValueError(f'Invalid metric "{item}". Use KEY=VALUE.')
        key, value = item.split('=', 1)
        parsed[key.strip()] = value.strip()
    return parsed


def kind_label(kind):
    return {
        'code': '代码',
        'experiment': '实验',
    }.get(kind, kind)


def metric_label(key):
    return {
        'F_score': 'F1',
        'recall': 'Recall',
        'precision': 'Precision',
        'cate_acc': '类别准确率',
        'category_accuracy': '类别准确率',
        'x_error_close': '近距离 x 误差',
        'x_error_far': '远距离 x 误差',
        'z_error_close': '近距离 z 误差',
        'z_error_far': '远距离 z 误差',
        'loss': 'Loss',
    }.get(key, key)


def display_title(entry):
    return entry.get('title_zh') or localize_text(entry.get('title', ''))


def display_summary(entry):
    return entry.get('summary_zh') or localize_text(entry.get('summary', ''))


def classify_module(file_path):
    path = file_path.replace('\\', '/')
    rules = (
        ('tools/research_journal.py', '实验记录工具'),
        ('tools/run_official_baselines.py', 'baseline 批量运行工具'),
        ('tools/train_dist.py', '分布式训练入口'),
        ('tools/train.py', '训练入口'),
        ('tools/test.py', '评测入口'),
        ('tools/deploy_test.py', '部署评测入口'),
        ('mmseg/models/lane_detector/', '车道检测器模块'),
        ('mmseg/models/losses/', '损失函数模块'),
        ('mmseg/models/utils/', '模型工具模块'),
        ('mmseg/datasets/pipelines/', '数据流水线模块'),
        ('mmseg/datasets/tools/', '数据集工具模块'),
        ('configs_v2/openlane/', 'OpenLane v2 配置'),
        ('configs/openlane/', 'OpenLane 配置'),
        ('configs/apollosim/', 'ApolloSim 配置'),
        ('BASELINE_REPRO_STATUS.md', 'baseline 复现台账'),
        ('V1_EXECUTION_PLAN.md', 'V1 执行规划'),
        ('V1_METHOD_DESIGN.md', 'V1 方法设计'),
    )
    for needle, label in rules:
        if needle in path:
            return label
    path_obj = Path(path)
    if len(path_obj.parts) >= 2:
        return '/'.join(path_obj.parts[-2:])
    return path


def unique_in_order(items):
    seen = set()
    ordered = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def infer_code_notes(entry):
    notes = []
    summary = display_summary(entry)
    if summary:
        notes.append(f'主要改动：{summary}')

    modules = unique_in_order(classify_module(path) for path in entry.get('files', []))
    if modules:
        notes.append('涉及模块：' + '、'.join(modules))

    if not notes:
        notes.append('本次代码记录未提供额外说明，请结合涉及文件查看改动。')
    return notes


def infer_experiment_notes(entry):
    notes = []
    summary = display_summary(entry)
    if summary:
        notes.append(summary)

    metrics = entry.get('metrics', {})
    if metrics:
        metric_text = '，'.join(f'{metric_label(k)}={v}' for k, v in metrics.items())
        notes.append('关键指标：' + metric_text)
    elif entry.get('exit_code') == 0 and entry.get('work_dir'):
        notes.append(f"输出目录：{entry['work_dir']}")
    elif entry.get('exit_code') not in (None, 0):
        notes.append(f"本次运行退出码为 {entry['exit_code']}，请结合日志继续排查。")

    if not notes:
        notes.append('本次实验记录未提供额外说明。')
    return notes


def render_markdown(entry):
    lines = [f"## {entry['timestamp']} | {kind_label(entry['kind'])} | {display_title(entry)}"]

    summary = display_summary(entry)
    if summary:
        lines.append(f"- 摘要：{summary}")

    if entry.get('tags'):
        lines.append(f"- 标签：{', '.join(entry['tags'])}")

    detail_key = 'module_notes' if entry['kind'] == 'code' else 'progress_notes'
    detail_prefix = '模块改动' if entry['kind'] == 'code' else '关键进展'
    details = entry.get(detail_key) or (
        infer_code_notes(entry) if entry['kind'] == 'code' else infer_experiment_notes(entry)
    )
    for idx, detail in enumerate(details, start=1):
        lines.append(f"- {detail_prefix} {idx}：{detail}")

    if entry.get('files'):
        lines.append(f"- 涉及文件：{', '.join(entry['files'])}")
    if entry.get('command'):
        lines.append(f"- 执行命令：`{entry['command']}`")
    if entry.get('work_dir'):
        lines.append(f"- 工作目录：`{entry['work_dir']}`")
    if entry.get('log_path'):
        lines.append(f"- 日志文件：`{entry['log_path']}`")
    if entry.get('exit_code') is not None:
        lines.append(f"- 退出码：{entry['exit_code']}")
    if entry.get('metrics'):
        metric_text = '，'.join(f'{metric_label(k)}={v}' for k, v in entry['metrics'].items())
        lines.append(f"- 指标：{metric_text}")

    git_head = entry.get('git', {}).get('head', '')
    git_branch = entry.get('git', {}).get('branch', '')
    if git_head or git_branch:
        short_head = git_head[:12] if git_head else ''
        lines.append(f"- Git 快照：`{git_branch}` @ `{short_head}`")

    lines.append('')
    return '\n'.join(lines)


def append_entry(journal_dir, entry):
    journal_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = journal_dir / 'journal.jsonl'
    markdown_path = journal_dir / 'journal.md'
    with jsonl_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    with markdown_path.open('a', encoding='utf-8') as f:
        f.write(render_markdown(entry))


def rebuild_markdown(journal_dir):
    journal_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = journal_dir / 'journal.jsonl'
    markdown_path = journal_dir / 'journal.md'
    entries = []
    if jsonl_path.exists():
        with jsonl_path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
    with markdown_path.open('w', encoding='utf-8') as f:
        for entry in entries:
            f.write(render_markdown(entry))


def handle_record_code(args):
    repo_root = Path(args.repo_root).resolve()
    journal_dir = Path(args.journal_dir).resolve()
    entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'kind': 'code',
        'title': args.title,
        'title_zh': args.title_zh,
        'summary': args.summary,
        'summary_zh': args.summary_zh,
        'tags': args.tag,
        'files': args.files,
        'module_notes': args.module_note,
        'git': git_snapshot(repo_root),
    }
    append_entry(journal_dir, entry)
    print(f'Code entry recorded in {journal_dir}')
    return 0


def handle_run_exp(args):
    repo_root = Path(args.repo_root).resolve()
    journal_dir = Path(args.journal_dir).resolve()
    log_dir = journal_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    command = list(args.command)
    if command and command[0] == '--':
        command = command[1:]
    if not command:
        raise ValueError('run-exp requires a command after "--".')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = log_dir / f'{timestamp}_{slugify(args.name)}.log'
    with log_path.open('w', encoding='utf-8') as log_file:
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        exit_code = process.wait()

    entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'kind': 'experiment',
        'title': args.name,
        'title_zh': args.title_zh,
        'summary': args.summary,
        'summary_zh': args.summary_zh,
        'tags': args.tag,
        'files': args.files,
        'progress_notes': args.progress_note,
        'command': shlex.join(command),
        'work_dir': args.work_dir,
        'log_path': str(log_path),
        'exit_code': exit_code,
        'metrics': parse_metrics(args.metric),
        'git': git_snapshot(repo_root),
    }
    append_entry(journal_dir, entry)
    print(f'Experiment entry recorded in {journal_dir}')
    return exit_code


def handle_rebuild_markdown(args):
    journal_dir = Path(args.journal_dir).resolve()
    rebuild_markdown(journal_dir)
    print(f'Markdown journal rebuilt in {journal_dir}')
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description='Record code changes and experiments for Anchor3dLane research.')
    parser.set_defaults(repo_root=str(default_repo_root()), journal_dir=str(default_journal_dir()))
    subparsers = parser.add_subparsers(dest='subcommand', required=True)

    record_code = subparsers.add_parser('record-code', help='Record a code change snapshot.')
    record_code.add_argument('--title', required=True)
    record_code.add_argument('--title-zh', default='')
    record_code.add_argument('--summary', required=True)
    record_code.add_argument('--summary-zh', default='')
    record_code.add_argument('--module-note', action='append', default=[])
    record_code.add_argument('--tag', action='append', default=[])
    record_code.add_argument('--files', nargs='*', default=[])
    record_code.add_argument('--repo-root', default=str(default_repo_root()))
    record_code.add_argument('--journal-dir', default=str(default_journal_dir()))
    record_code.set_defaults(handler=handle_record_code)

    run_exp = subparsers.add_parser('run-exp', help='Run an experiment command and record the result.')
    run_exp.add_argument('--name', required=True)
    run_exp.add_argument('--title-zh', default='')
    run_exp.add_argument('--summary', default='')
    run_exp.add_argument('--summary-zh', default='')
    run_exp.add_argument('--progress-note', action='append', default=[])
    run_exp.add_argument('--tag', action='append', default=[])
    run_exp.add_argument('--files', nargs='*', default=[])
    run_exp.add_argument('--work-dir', default='')
    run_exp.add_argument('--metric', action='append', default=[])
    run_exp.add_argument('--repo-root', default=str(default_repo_root()))
    run_exp.add_argument('--journal-dir', default=str(default_journal_dir()))
    run_exp.add_argument('command', nargs=argparse.REMAINDER)
    run_exp.set_defaults(handler=handle_run_exp)

    rebuild = subparsers.add_parser('rebuild-markdown', help='Rebuild journal.md from journal.jsonl.')
    rebuild.add_argument('--journal-dir', default=str(default_journal_dir()))
    rebuild.set_defaults(handler=handle_rebuild_markdown)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    return args.handler(args)


if __name__ == '__main__':
    raise SystemExit(main())
