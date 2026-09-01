import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { TerminalProposal } from "./App";

afterEach(() => cleanup());

describe("TerminalProposal", () => {
  it("requires an explicit click and runs the edited command", async () => {
    const user = userEvent.setup();
    const onRun = vi.fn();
    render(
      <TerminalProposal
        proposalId="proposal_1"
        command="pytest -q"
        reason="Inspect failing tests"
        onRun={onRun}
      />,
    );

    expect(onRun).not.toHaveBeenCalled();
    const editor = screen.getByLabelText("Command");
    await user.clear(editor);
    await user.type(editor, "python -m unittest");
    await user.click(screen.getByRole("button", { name: "Run" }));

    expect(onRun).toHaveBeenCalledWith("proposal_1", "python -m unittest");
  });

  it("can cancel without executing", async () => {
    const user = userEvent.setup();
    const onRun = vi.fn();
    render(
      <TerminalProposal
        proposalId="proposal_2"
        command="git status"
        reason="Inspect the repository"
        onRun={onRun}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(screen.queryByLabelText("Proposed terminal command")).not.toBeInTheDocument();
    expect(onRun).not.toHaveBeenCalled();
  });
});
