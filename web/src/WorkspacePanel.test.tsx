import { StrictMode } from "react";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

const workspaceApiMocks = vi.hoisted(() => ({
  createWorkspaceTerminal: vi.fn(),
  getWorkspaceStatus: vi.fn(),
  listWorkspaceTerminals: vi.fn(),
}));

vi.mock("./api", async (importOriginal) => ({
  ...await importOriginal<typeof import("./api")>(),
  ...workspaceApiMocks,
}));

import { TerminalProposal } from "./App";
import WorkspacePanel from "./WorkspacePanel";

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

describe("WorkspacePanel", () => {
  it("handles each command request only once in StrictMode", async () => {
    workspaceApiMocks.getWorkspaceStatus.mockResolvedValue({
      available: true,
      status: "running",
      chat_id: "chat_1",
    });
    workspaceApiMocks.listWorkspaceTerminals.mockResolvedValue([]);
    workspaceApiMocks.createWorkspaceTerminal.mockReturnValue(new Promise(() => {}));

    render(
      <StrictMode>
        <WorkspacePanel
          chatId="chat_1"
          visible
          onVisibleChange={vi.fn()}
          commandRequest={{ id: 1, command: "python report.py" }}
        />
      </StrictMode>,
    );

    await waitFor(() => expect(workspaceApiMocks.createWorkspaceTerminal).toHaveBeenCalledTimes(1));
    expect(workspaceApiMocks.createWorkspaceTerminal).toHaveBeenCalledWith("chat_1");
  });
});
