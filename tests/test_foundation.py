from main import main


def test_main_runs(capsys):
    main()

    captured = capsys.readouterr()

    assert "TalkToTableau" in captured.out
    assert "Project foundation is working." in captured.out