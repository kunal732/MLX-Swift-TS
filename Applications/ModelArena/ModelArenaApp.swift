import SwiftUI

@main
struct ModelArenaApp: App {
    @State private var runner = ArenaRunner()
    @State private var vm = ArenaVM()
    @State private var customHubID = ""
    @State private var showAddModel = false

    #if os(iOS)
    @State private var showDocumentPicker = false
    #endif

    var body: some Scene {
        WindowGroup {
            VStack(spacing: 0) {
                #if os(macOS)
                macOSToolbar
                #else
                iOSToolbar
                #endif

                Divider()

                ArenaView(vm: vm, runner: runner)
            }
            #if os(iOS)
            .sheet(isPresented: $showDocumentPicker) {
                DocumentPickerView { url in
                    runner.loadFromDirectory(url)
                }
            }
            #endif
        }
        #if os(macOS)
        .defaultSize(width: 1200, height: 800)
        #endif
    }

    // MARK: - Add Model Popover (shared)

    private var addModelPopover: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Add HuggingFace Model")
                .font(.subheadline)
                .fontWeight(.semibold)

            TextField("e.g. Datadog/Toto-Open-Base-1.0", text: $customHubID)
                .textFieldStyle(.roundedBorder)
                #if os(macOS)
                .frame(width: 300)
                #endif

            HStack {
                Spacer()
                Button("Add") {
                    let id = customHubID.trimmingCharacters(in: .whitespaces)
                    guard !id.isEmpty else { return }
                    let name = id.components(separatedBy: "/").last ?? id
                    showAddModel = false
                    Task {
                        await runner.addFromHub(id: id, name: name)
                    }
                    customHubID = ""
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .disabled(customHubID.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
        .padding()
    }

    #if os(macOS)
    // MARK: - macOS Toolbar

    private var macOSToolbar: some View {
        HStack(spacing: 12) {
            Image(systemName: "trophy")
                .foregroundStyle(runner.loadedCount > 0 ? .yellow : .secondary)

            Text("Model Arena")
                .font(.headline)

            Spacer()

            if runner.isLoading {
                ProgressView(value: runner.loadProgress > 0 ? runner.loadProgress : nil)
                    .frame(width: 120)
                Text(runner.loadingStatus)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            } else if runner.loadedCount > 0 {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                Text("\(runner.loadedCount) model(s) ready")
                    .font(.caption)
                    .foregroundStyle(.green)
            } else {
                Text("No models loaded")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Button {
                showAddModel.toggle()
            } label: {
                Image(systemName: "plus")
            }
            .controlSize(.small)
            .disabled(runner.isLoading)
            .popover(isPresented: $showAddModel) { addModelPopover }

            Button("Load from Directory") {
                let panel = NSOpenPanel()
                panel.title = "Select Models Directory"
                panel.message = "Choose a directory containing converted model subdirectories"
                panel.canChooseDirectories = true
                panel.canChooseFiles = false
                panel.allowsMultipleSelection = false

                if panel.runModal() == .OK, let url = panel.url {
                    runner.loadFromDirectory(url)
                }
            }
            .controlSize(.small)
            .disabled(runner.isLoading)

            Button("Load from HF") {
                Task {
                    await runner.loadFromHub(models: defaultHubModels)
                }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .disabled(runner.isLoading)
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(.bar)
    }
    #endif

    #if os(iOS)
    // MARK: - iOS Toolbar

    private var iOSToolbar: some View {
        VStack(spacing: 6) {
            // Row 1: Title + status
            HStack(spacing: 8) {
                Image(systemName: "trophy")
                    .foregroundStyle(runner.loadedCount > 0 ? .yellow : .secondary)

                Text("Model Arena")
                    .font(.headline)

                Spacer()

                if runner.isLoading {
                    ProgressView(value: runner.loadProgress > 0 ? runner.loadProgress : nil)
                        .frame(width: 80)
                    Text(runner.loadingStatus)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .frame(maxWidth: 100)
                } else if runner.loadedCount > 0 {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                    Text("\(runner.loadedCount) ready")
                        .font(.caption)
                        .foregroundStyle(.green)
                } else {
                    Text("No models")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            // Row 2: Action buttons
            HStack(spacing: 8) {
                Button {
                    showAddModel.toggle()
                } label: {
                    Image(systemName: "plus.circle")
                }
                .disabled(runner.isLoading)
                .popover(isPresented: $showAddModel) { addModelPopover }

                Button {
                    showDocumentPicker = true
                } label: {
                    Label("Files", systemImage: "folder")
                        .font(.caption)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(runner.isLoading)

                Button {
                    Task {
                        await runner.loadFromHub(models: defaultHubModels)
                    }
                } label: {
                    Label("HuggingFace", systemImage: "arrow.down.circle")
                        .font(.caption)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .disabled(runner.isLoading)

                Spacer()
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 6)
        .background(.bar)
    }
    #endif
}
