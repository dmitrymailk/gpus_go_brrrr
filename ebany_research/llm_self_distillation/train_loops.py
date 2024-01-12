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
        total_losses_dict = {
            f"total_train_loss_step_{i}": 0
            for i, item in enumerate(model.gpt_neox.layers)
        }

        active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                # outputs = model(**batch, layer_iterations=layer_iterations)
                outputs = model(
                    global_step=completed_steps,
                    **batch,
                )
                lm_loss = outputs["loss"]
                all_losses = outputs["all_losses"]
                loss = all_losses[-1]
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss
                    # losses_dict = {
                    #     f"train_loss_step_{i}": item
                    #     for i, item in range(
                    #         len(model.gpt_neox.layers) - 1 - model.classifiers_amount,
                    #         len(model.gpt_neox.layers) - 1,
                    #     )
                    # }
                    losses_dict = {}
                    loss_pos = 0
                    for layer_pos in range(len(model.gpt_neox.layers)):
                        # if (
                        #     layer_pos
                        #     < len(model.gpt_neox.layers) - 1 - model.classifiers_amount
                        # ):
                        #     losses_dict[f"train_loss_step_{layer_pos}"] = 0.0
                        #     total_losses_dict[
                        #         f"total_train_loss_step_{layer_pos}"
                        #     ] = 0.0
                        # else:
                        #     losses_dict[f"train_loss_step_{layer_pos}"] = outputs[
                        #         "all_losses"
                        #     ][loss_pos]
                        #     total_losses_dict[
                        #         f"total_train_loss_step_{layer_pos}"
                        #     ] += outputs["all_losses"][loss_pos]
                        #     loss_pos += 1
                        if layer_pos > model.classifiers_amount - 1:
                            losses_dict[f"train_loss_step_{layer_pos}"] = 0.0
                            total_losses_dict[
                                f"total_train_loss_step_{layer_pos}"
                            ] = 0.0
                        else:
                            losses_dict[f"train_loss_step_{layer_pos}"] = outputs[
                                "all_losses"
                            ][loss_pos]
                            total_losses_dict[
                                f"total_train_loss_step_{layer_pos}"
                            ] += outputs["all_losses"][loss_pos]
                            loss_pos += 1

                    accelerator.log(
                        {
                            "train_loss_step": loss,
                            **losses_dict,
                        },
                        step=completed_steps,
                    )
                accelerator.backward(lm_loss)
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
                outputs = model(
                    global_step=completed_steps,
                    **batch,
                )

            loss = outputs["all_losses"][-1]
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
            for key in list(total_losses_dict.keys()):
                total_losses_dict[key] = total_losses_dict[key] / len(train_dataloader)
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                    **total_losses_dict,
                },
                step=completed_steps,
            )
