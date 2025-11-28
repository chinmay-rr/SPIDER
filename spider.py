

from typing import List, Optional, Callable

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, _use_grad_for_differentiable

__all__ = ["Spider", "spider", "make_closure"]


def make_closure(model, loss_fn, batch):
    inputs, targets = batch

    def closure():
        model.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        return loss

    return closure


class Spider(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        freq: int = 100,
        maximize: bool = False,
        foreach: Optional[bool] = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if freq < 1:
            raise ValueError(f"Invalid frequency value: {freq}")

        defaults = dict(
            lr=lr,
            freq=freq,
            maximize=maximize,
            foreach=foreach,
            differentiable=False,
        )

        super().__init__(params, defaults)

        # Global step counter for the optimizer 
        self.state.setdefault("_step", 0)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("differentiable", False)

        self.state.setdefault("_step", 0)

    @_use_grad_for_differentiable
    def step(self, closure: Callable = None, full_grad_closure: Optional[Callable] = None):
        
        if closure is None:
            raise RuntimeError("Spider requires a closure argument (which must reuse the same batch).")

        # Use a global step counter 
        step_count = self.state.get("_step", 0)
        update_frequency = self.param_groups[0]["freq"]

        loss = None
        if step_count % update_frequency == 0:
            if full_grad_closure is None:
                raise RuntimeError(
                    f"Step {step_count} requires a full_grad_closure to reset Spider estimator."
                )

            # Compute gradients on the large batch / full dataset.
            with torch.enable_grad():
                self.zero_grad()
                loss = full_grad_closure()

            # Initialize / reset the estimator (v_t) and prev_param buffer
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    state["estimator"] = p.grad.detach().clone()
                
                    state["prev_param"] = p.detach().clone().to(p.device).to(p.dtype)
                    with torch.no_grad():
                        if group["maximize"]:
                            p.add_(state["estimator"], alpha=group["lr"])
                        else:
                            p.add_(state["estimator"], alpha=-group["lr"])

            # Increment global step and return
            self.state["_step"] = step_count + 1
            return loss


        with torch.enable_grad():
            self.zero_grad()
            loss = closure()

        current_grads_buffer: List[List[Optional[Tensor]]] = []
        for group in self.param_groups:
            group_grads: List[Optional[Tensor]] = []
            for p in group["params"]:
                if p.grad is not None:
                    group_grads.append(p.grad.detach().clone())
                else:
                    group_grads.append(None)
            current_grads_buffer.append(group_grads)

        # Swap weights: load x_{t-1} into the model 
        current_params_buffer: List[Tensor] = []
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p, {})
                if "prev_param" in state:
                    # Backup current param 
                    current_params_buffer.append(p.data.clone())
                    with torch.no_grad():
                        p.data.copy_(state["prev_param"])

        #  Calculate gradient at previous weights
        with torch.enable_grad():
            self.zero_grad()
            loss_prev = closure()  # explicit capture for clarity

        # Swap weights back
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p, {})
                if "prev_param" in state:
                    with torch.no_grad():
                        p.data.copy_(current_params_buffer[idx])
                    idx += 1

        # 5. Build lists and call functional update
        for i, group in enumerate(self.param_groups):
            params_list: List[Tensor] = []
            grads_prev_list: List[Tensor] = []  # ∇f_S(x_{t-1}) 
            grads_curr_list: List[Tensor] = []  # ∇f_S(x_t) (from buffer)
            estimator_list: List[Tensor] = []   # v_{t-1} (in state)
            prev_params_list: List[Tensor] = [] # buffers to update to x_t for next step
            # above are sent to spider function which does the updating

            current_group_grads = current_grads_buffer[i]

            for j, p in enumerate(group["params"]):
                # p.grad currently holds gradient computed at x_{t-1}
                if p.grad is not None and current_group_grads[j] is not None:
                    state = self.state[p]

                    params_list.append(p)
                    # Always use detached clones for grads_prev to be safe
                    grads_prev_list.append(p.grad.detach().clone())
                    grads_curr_list.append(current_group_grads[j])
                    estimator_list.append(state["estimator"])

                    # Ensure prev_param exists and has correct device/dtype
                    if "prev_param" not in state:
                        state["prev_param"] = p.detach().clone().to(p.device).to(p.dtype)
                    prev_params_list.append(state["prev_param"])

            # Only run spider if there is something to update
            if len(params_list) > 0:
                spider(
                    params_list,
                    grads_curr_list,
                    grads_prev_list,
                    estimator_list,
                    prev_params_list,
                    lr=group["lr"],
                    maximize=group["maximize"],
                    foreach=group["foreach"],
                )

        # Increment global step and return the loss on x_t
        self.state["_step"] = step_count + 1
        return loss

def spider(
    params: List[Tensor],
    grads_curr: List[Tensor],
    grads_prev: List[Tensor],
    estimators: List[Tensor],
    prev_params: List[Tensor],
    *,
    lr: float,
    maximize: bool,
    foreach: Optional[bool] = None,
):
   
    use_foreach = False if foreach is None else foreach

    # Quick capability check: don't try to use private/experimental foreach ops if not available
    if use_foreach:
        try:
            # Check presence of the core ops we rely on
            _ = torch._foreach_add_
            _ = torch._foreach_sub
            _ = torch._foreach_copy_
            _ = torch._foreach_add  # reading presence; may raise if not present
        except Exception:
            use_foreach = False

    if use_foreach:
        _multi_tensor_spider(
            params,
            grads_curr,
            grads_prev,
            estimators,
            prev_params,
            lr=lr,
            maximize=maximize,
        )
    else:
        _single_tensor_spider(
            params,
            grads_curr,
            grads_prev,
            estimators,
            prev_params,
            lr=lr,
            maximize=maximize,
        )


def _single_tensor_spider(
    params: List[Tensor],
    grads_curr: List[Tensor],
    grads_prev: List[Tensor],
    estimators: List[Tensor],
    prev_params: List[Tensor],
    *,
    lr: float,
    maximize: bool,
):
    for i, param in enumerate(params):
        g_curr = grads_curr[i]
        g_prev = grads_prev[i]
        v_old = estimators[i]
        x_prev_buffer = prev_params[i]
        if maximize:
            # flip signs for maximize 
            g_curr = -g_curr
            g_prev = -g_prev

        # v_t = (g_curr - g_prev) + v_{t-1}
        diff = g_curr.sub(g_prev)  
        v_old.add_(diff)

        # Update previous param buffer (save current parameter for next step)
        with torch.no_grad():
            x_prev_buffer.copy_(param)

        # 3. Update parameters: x = x - lr * v
        with torch.no_grad():
            if maximize:
                param.add_(v_old, alpha=lr)  # v_old is now v_new
            else:
                param.add_(v_old, alpha=-lr)


def _multi_tensor_spider(
    params: List[Tensor],
    grads_curr: List[Tensor],
    grads_prev: List[Tensor],
    estimators: List[Tensor],
    prev_params: List[Tensor],
    *,
    lr: float,
    maximize: bool,
):
    if len(params) == 0:
        return

    # Group tensors by device/dtype for foreach ops
    grouped = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads_curr, grads_prev, estimators, prev_params]
    )

    for (
        device_params,
        device_grads_curr,
        device_grads_prev,
        device_estimators,
        device_prev_params,
    ), _ in grouped.values():
        # device_* lists contain tensors already matching device/dtype

        # Optionally negate grads for maximize
        if maximize:
            # Use foreach negation if available, otherwise elementwise
            try:
                device_grads_curr = torch._foreach_neg(device_grads_curr)
                device_grads_prev = torch._foreach_neg(device_grads_prev)
            except Exception:
                device_grads_curr = [ -g for g in device_grads_curr ]
                device_grads_prev = [ -g for g in device_grads_prev ]

        # diff = g_curr - g_prev (foreach)
        try:
            diff = torch._foreach_sub(device_grads_curr, device_grads_prev)
        except Exception:
            diff = [a - b for a, b in zip(device_grads_curr, device_grads_prev)]

        # v = v + diff (in-place foreach add)
        try:
            torch._foreach_add_(device_estimators, diff)
        except Exception:
            for v, d in zip(device_estimators, diff):
                v.add_(d)

        # x_prev <- x_curr (copy current param values into prev_param buffers)
        try:
            torch._foreach_copy_(device_prev_params, device_params)
        except Exception:
            for prev_buf, p in zip(device_prev_params, device_params):
                prev_buf.copy_(p)

        #  x = x - lr * v
        alpha = lr if maximize else -lr
        try:
            torch._foreach_add_(device_params, device_estimators, alpha=alpha)
        except Exception:
            for p, v in zip(device_params, device_estimators):
                p.add_(v, alpha=alpha)
