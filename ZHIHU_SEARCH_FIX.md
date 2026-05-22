# Fix for zhihu_search and other search commands

## Problem

The `zhihu_search` tool was failing with the error:
```
error: required option '--keyword <value>' not specified
```

This happened because the code was hardcoded to use `--query` for all search commands, but some sites (zhihu, bilibili, xiaohongshu, smzdm) use `--keyword` instead.

## Root Cause

In `deepfind/tools.py`, the `_opencli_site_search` method at line 1440 was hardcoded to use `--query`:

```python
command.extend(["--query", query])
```

However, the zhihu search command expects `--keyword`:
```bash
opencli zhihu search --keyword <value>
```

## Solution

Modified the `_opencli_site_search` method to dynamically detect the correct parameter name:

```python
# Some sites use "keyword" instead of "query" (e.g., zhihu)
query_param = "keyword" if self._opencli_supports_arg(command_spec, "keyword") else "query"
if self._opencli_arg_is_positional(command_spec, query_param):
    command.append(query)
else:
    command.extend([f"--{query_param}", query])
```

## Impact

This fix resolves the issue for all sites that use `--keyword`:
- zhihu/search ✅
- bilibili/search ✅
- xiaohongshu/search ✅
- smzdm/search ✅

Sites that use `--query` continue to work as before:
- boss/search
- coupang/search
- ctrip/search
- linkedin/search
- reddit/search
- reuters/search
- twitter/search
- xueqiu/search
- youtube/search

## Testing

Verified that the parameter detection logic correctly identifies "keyword" when available and falls back to "query" otherwise.

## File Changed

- `deepfind/tools.py` (lines 1436-1445)
