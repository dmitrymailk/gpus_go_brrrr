import torch
import math


def experiment_1(
    starting_epoch=None,
    args=None,
    model=None,
    train_dataloader=None,
    accelerator=None,
    optimizer=None,
    lr_scheduler=None,
    progress_bar=None,
    eval_dataloader=None,
    logger=None,
    completed_steps=None,
):
    layer_iterations = 100
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0

        active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                # outputs = model(**batch, layer_iterations=layer_iterations)
                outputs = model(
                    **batch,
                )
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                    accelerator.log(
                        {
                            "train_loss_step": loss.detach().float(),
                        },
                        step=completed_steps,
                    )
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(
                accelerator.gather_for_metrics(
                    loss.repeat(args.per_device_eval_batch_size)
                )
            )

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        if args.with_tracking:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )


def experiment_2(
    starting_epoch=None,
    args=None,
    model=None,
    train_dataloader=None,
    accelerator=None,
    optimizer=None,
    lr_scheduler=None,
    progress_bar=None,
    eval_dataloader=None,
    logger=None,
    completed_steps=None,
):
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0

        active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            layer_iterations = 1
            if epoch == 0:
                layer_iterations = 1
            elif epoch == 1:
                layer_iterations = 2

            with accelerator.accumulate(model):
                outputs = model(**batch, layer_iterations=layer_iterations)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                    accelerator.log(
                        {
                            "train_loss_step": loss.detach().float(),
                        },
                        step=completed_steps,
                    )
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(
                accelerator.gather_for_metrics(
                    loss.repeat(args.per_device_eval_batch_size)
                )
            )

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        if args.with_tracking:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if epoch == 0:
            model.gpt_neox.layers[1].load_state_dict(
                model.gpt_neox.layers[0].state_dict()
            )


def experiment_3(
    starting_epoch=None,
    args=None,
    model=None,
    train_dataloader=None,
    accelerator=None,
    optimizer=None,
    lr_scheduler=None,
    progress_bar=None,
    eval_dataloader=None,
    logger=None,
    completed_steps=None,
):
    copy_step = 2
    layer_iterations = 0
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0

        active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch, layer_iterations=layer_iterations)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                    accelerator.log(
                        {
                            "train_loss_step": loss.detach().float(),
                        },
                        step=completed_steps,
                    )
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

            if epoch == 0:
                if step == copy_step * layer_iterations:
                    layer_iterations += 1
                    if layer_iterations < len(model.gpt_neox.layers) :
                        print(f"step={step} layer_iterations={layer_iterations}")
                        model.gpt_neox.layers[layer_iterations].load_state_dict(
                            model.gpt_neox.layers[layer_iterations - 1].state_dict()
                        )

        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(
                accelerator.gather_for_metrics(
                    loss.repeat(args.per_device_eval_batch_size)
                )
            )

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        if args.with_tracking:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )
