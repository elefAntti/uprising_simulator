import argparse
import io
import zipfile
from collections import OrderedDict

import torch

EXPECTED_OBS = 134
EXPECTED_H1 = 128
EXPECTED_H2 = 128
EXPECTED_ACT = 2

def main():
    ap = argparse.ArgumentParser(description="Convert SB3 PPO (ppo_policy.zip) → BC-style state_dict (bc_policy.pth)")
    ap.add_argument("--sb3-zip", default="ppo_policy.zip", help="Path to SB3 policy zip (from model.save)")
    ap.add_argument("--out", default="bc_policy.pth", help="Output .pth for ClonedBot")
    ap.add_argument("--strict", action="store_true", help="Fail if sizes differ from 134-128-128-2")
    args = ap.parse_args()

    # 1) Load raw policy state_dict from the SB3 zip (no cloudpickle)
    with zipfile.ZipFile(args.sb3_zip, "r") as zf:
        if "policy.pth" not in zf.namelist():
            raise FileNotFoundError("policy.pth not found inside zip — is this an SB3 save?")
        with zf.open("policy.pth", "r") as f:
            sd = torch.load(io.BytesIO(f.read()), map_location="cpu")

    def get_shape(key):
        t = sd.get(key)
        if t is None:
            raise KeyError(f"Missing key in SB3 state_dict: {key}")
        return t.shape

    # 2) Inspect sizes
    w0_shape = get_shape("mlp_extractor.policy_net.0.weight")   # (h1, obs)
    b0_shape = get_shape("mlp_extractor.policy_net.0.bias")     # (h1,)
    w2_shape = get_shape("mlp_extractor.policy_net.2.weight")   # (h2, h1)
    b2_shape = get_shape("mlp_extractor.policy_net.2.bias")     # (h2,)
    wa_shape = get_shape("action_net.weight")                   # (act, h2)
    ba_shape = get_shape("action_net.bias")                     # (act,)

    obs_dim = w0_shape[1]
    h1 = w0_shape[0]
    h2 = w2_shape[0]
    act_dim = wa_shape[0]

    msg = f"Detected SB3 policy sizes: obs={obs_dim}, h1={h1}, h2={h2}, act={act_dim}"
    print(msg)

    if args.strict:
        exp = (EXPECTED_OBS, EXPECTED_H1, EXPECTED_H2, EXPECTED_ACT)
        got = (obs_dim, h1, h2, act_dim)
        if got != exp:
            raise RuntimeError(
                f"Size mismatch (strict mode): expected {exp}, got {got}. "
                f"Adjust your SB3 policy net_arch or disable --strict."
            )

    # 3) Map SB3 policy weights → BC-style keys that ClonedBot expects
    # ClonedBot BCNet is nn.Sequential([...Linear(obs,128), ReLU, Linear(128,128), ReLU, Linear(128,2), Tanh])
    out_sd = OrderedDict()
    out_sd["network.0.weight"] = sd["mlp_extractor.policy_net.0.weight"].detach().cpu()
    out_sd["network.0.bias"]   = sd["mlp_extractor.policy_net.0.bias"].detach().cpu()
    out_sd["network.2.weight"] = sd["mlp_extractor.policy_net.2.weight"].detach().cpu()
    out_sd["network.2.bias"]   = sd["mlp_extractor.policy_net.2.bias"].detach().cpu()
    out_sd["network.4.weight"] = sd["action_net.weight"].detach().cpu()
    out_sd["network.4.bias"]   = sd["action_net.bias"].detach().cpu()

    # 4) Save to the BC file
    torch.save(out_sd, args.out)
    print(f"Saved BC-style weights to: {args.out}")

if __name__ == "__main__":
    main()

