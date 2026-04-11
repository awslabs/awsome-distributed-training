"""Patch HybridDeviceOptimizer._sync_hdo_param_groups_to_sub_optimizers.

Root cause (Memory #118): Float16OptimizerWithFloat16Params.__init__ replaces
BF16 params in HybridDeviceOptimizer's param_groups with FP32 master copies.
But HDO's param_to_inner_param dict was built against the original BF16 params,
so _sync_hdo_param_groups_to_sub_optimizers() gets KeyError when iterating
self.param_groups (now containing FP32 masters) and looking them up.

Fix: Use position-based group matching instead of param identity matching.
"""
import re
import sys

filepath = (
    '/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM/'
    'megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py'
)

with open(filepath, 'r') as f:
    content = f.read()

new_method = '''    def _sync_hdo_param_groups_to_sub_optimizers(self):
        """Sync HDO new param_groups attribute (e.g. lr, wd, etc.) to sub-optimizers.

        PATCHED: Use group position matching instead of param identity matching.
        Float16OptimizerWithFloat16Params replaces BF16 params with FP32 masters
        in self.param_groups, breaking param_to_inner_param identity lookups.
        Since all sub-optimizer groups inherit the same hyperparams (lr, wd, etc.)
        from the HDO, we broadcast from the first HDO group to all sub-optimizer groups.
        """
        # Get the non-params attributes from each HDO param_group
        hdo_group_attrs = []
        for group in self.param_groups:
            attrs = {k: v for k, v in group.items() if k != "params"}
            hdo_group_attrs.append(attrs)

        for optimizer in self.sub_optimizers:
            new_param_groups = []
            for group in optimizer.param_groups:
                new_group = group.copy()
                assert len(group["params"]) > 0, "param_groups should not be empty"
                # Use the first HDO group attrs as default (all groups typically share
                # the same lr/wd in NeMo RL GRPO). If we have matching group count,
                # use positional matching.
                if len(hdo_group_attrs) == len(optimizer.param_groups):
                    update_attrs = hdo_group_attrs[optimizer.param_groups.index(group)]
                elif len(hdo_group_attrs) > 0:
                    update_attrs = hdo_group_attrs[0]
                else:
                    update_attrs = {}
                new_group.update(update_attrs)
                new_param_groups.append(new_group)
            optimizer.param_groups = new_param_groups'''

# Strategy: regex replace the entire method definition up to the next method
pattern = (
    r'(    def _sync_hdo_param_groups_to_sub_optimizers\(self\):.*?)'
    r'(\n    def )'
)
match = re.search(pattern, content, re.DOTALL)
if match:
    content = content[:match.start()] + new_method + '\n\n' + '    def ' + content[match.end():]
    with open(filepath, 'w') as f:
        f.write(content)
    print('PATCHED: _sync_hdo_param_groups_to_sub_optimizers (position-based sync)')
else:
    print('ERROR: Could not find method to patch', file=sys.stderr)
    sys.exit(1)

# Delete stale .pyc cache and recompile
import py_compile
import glob
import os

pyc_dir = os.path.join(os.path.dirname(filepath), '__pycache__')
if os.path.isdir(pyc_dir):
    for pyc in glob.glob(os.path.join(pyc_dir, 'hybrid_optimizer*.pyc')):
        os.remove(pyc)
        print(f'Removed stale .pyc: {pyc}')

py_compile.compile(filepath, doraise=True)
print(f'Recompiled: {filepath}')
