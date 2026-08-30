// Copyright 2026 Lars Brubaker
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// FileComplianceTests.cs — NOT A PORT. This file has no counterpart in
// manifold-rust; it is a C#-only house-rules test that makes CLAUDE.md's
// "File-size rule" section executable, adapted from MatterCAD's
// Tests/MatterCADTests/Standard/FileComplianceTests.cs (same 800 limit, same
// per-file exemption dictionary, same violation-reporting style).
//
// ── One deliberate divergence from the MatterCAD original: the metric ────────
// MatterCAD counts non-empty lines; this counts total lines. The cap here is
// inherited from the Rust port, and every number written about it is a total:
// docs/PORTING_PLAN.md's "800-line file cap", the Rust files' own exemptions, and
// QuickHull.Algo.cs's header arithmetic ("60 lines over the 800-line cap").
// Counting the same thing keeps a C# file's number directly comparable to the
// Rust file it was ported from, which is the whole point of having one number.
//
// The cost of that choice is honest: under a total-line metric, deleting blank
// lines *would* lower the count. CLAUDE.md forbids it, and this test cannot
// detect the intent behind an edit (deleting comments defeats either metric
// equally). That clause is enforced at review, not here.
//
// Nothing in here asserts anything about geometry, so it is exempt from the
// "expected values identical to Rust" bar the rest of the suite lives under.

using System.Runtime.CompilerServices;
using System.Text;

using TUnit.Assertions;
using TUnit.Assertions.Enums;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	/// <summary>
	/// Tests that all source files in the repo conform to the file size limit.
	/// File size is measured as total lines. Default limit: 800 lines; individual files
	/// may carry a documented higher limit in <see cref="ExplicitFileLimits"/>.
	/// </summary>
	public class FileComplianceTests
	{
		/// <summary>
		/// Default maximum lines for any source file.
		/// </summary>
		private const int DefaultLineLimit = 800;

		/// <summary>
		/// The file that marks the repository root, used to resolve it at runtime.
		/// </summary>
		private const string RepoMarkerFile = "ManifoldSharp.sln";

		/// <summary>
		/// Explicit file size limits for specific exempted files.
		/// Paths are relative to the repo root using forward slashes.
		/// </summary>
		/// <remarks>
		/// From CLAUDE.md's "File-size rule": the 800-line file cap is enforced by this
		/// test, with an explicit exemption list (the Rust exemptions — linalg, edge_op,
		/// quickhull_algo — were earned by *their* coupling; the C# files split instead
		/// wherever they could). Adding an entry requires the same documented
		/// justification the Rust files carry, written in the exempted file's own header
		/// where a reader meets it, not only here. Reducing line count by deleting
		/// comments or blank lines is not compliance — this port's file headers and
		/// invariant comments are the ported artifact, not filler.
		/// <para>
		/// Each ceiling sits a little above the file's current size, so the exemption
		/// covers the file as written but ordinary growth still trips this test.
		/// Ceilings should only ever decrease, never increase; remove an entry entirely
		/// once its file is back under the default.
		/// </para>
		/// </remarks>
		private static readonly Dictionary<string, int> ExplicitFileLimits = new Dictionary<string, int>
		{
			// The one file that takes the quickhull_algo exemption CLAUDE.md grants. Its own
			// header carries the justification: the only remaining cut is between
			// SetupInitialTetrahedron and CreateConvexHalfedgeMesh, and those two are one
			// argument — the degenerate branches in the first (single point, 1D, planar) are
			// exactly what the second's loop is allowed to assume away, and neither is
			// checkable without the other in view. 867 lines as written; the ceiling is
			// 880 so the file can be edited without churn here, but not grown into.
			["ManifoldSharp/QuickHull.Algo.cs"] = 880,
		};

		/// <summary>
		/// The project directories scanned, relative to the repo root.
		/// </summary>
		private static readonly string[] ScannedProjectDirectories =
		{
			"ManifoldSharp",
			"ManifoldSharp.Tests",
			"ManifoldSharp.OracleTests",
			"ManifoldSharp.Benchmarks",
		};

		/// <summary>
		/// Directory names to exclude from scanning, anywhere in the tree.
		/// </summary>
		private static readonly HashSet<string> ExcludedDirectories = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
		{
			"bin",
			"obj",
			".git",
			".vs",
			".vscode",
			"node_modules",
			"TestResults",
		};

		/// <summary>
		/// File extensions to include in scanning.
		/// </summary>
		private static readonly HashSet<string> IncludedExtensions = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
		{
			".cs",
		};

		[Test]
		public async Task AllFilesShouldComplyWithSizeLimits()
		{
			string repoRoot = ResolveRepoRoot();
			List<string> files = GetAllProjectFiles(repoRoot);
			List<string> violations = new List<string>();

			foreach (string filePath in files)
			{
				int lineCount = CountLines(filePath);
				string relativePath = GetRelativePath(repoRoot, filePath);
				int limit = GetFileLimit(relativePath);

				if (lineCount > limit)
				{
					violations.Add($"  {relativePath}: {lineCount} lines (limit: {limit}) - this must be refactored into multiple smaller files.");
				}
			}

			if (violations.Count > 0)
			{
				StringBuilder message = new StringBuilder();
				message.AppendLine($"File size violations found ({violations.Count} files exceed their limits):");
				message.AppendLine();
				foreach (string violation in violations.OrderByDescending(v => v))
				{
					message.AppendLine(violation);
				}

				message.AppendLine();
				message.AppendLine("To fix: split the file into smaller, cohesive modules, the way the rest of");
				message.AppendLine("this port does (ManifoldImpl.cs / .Topology.cs / .Shapes.cs, EdgeOp.cs and its");
				message.AppendLine("four companions). Carry the module header on the primary file and point the");
				message.AppendLine("others back at it.");
				message.AppendLine("Stripping comments or blank lines to get under the number is NOT compliance");
				message.AppendLine("- see CLAUDE.md, \"File-size rule\". An exemption in ExplicitFileLimits needs");
				message.AppendLine("the same written justification the Rust port's own exemptions carry, stated");
				message.AppendLine("in the exempted file's own header (QuickHull.Algo.cs is the worked example).");

				Assert.Fail(message.ToString());
			}

			// Also the canary: a scan that matched nothing would report zero violations and
			// look identical to a clean tree.
			await Assert.That(files.Count).IsGreaterThan(0);
		}

		[Test]
		public async Task ComplianceSummaryReport()
		{
			string repoRoot = ResolveRepoRoot();
			List<string> files = GetAllProjectFiles(repoRoot);

			List<(string Path, int Lines, int Limit)> violations = new List<(string, int, int)>();
			int largest = 0;
			string largestPath = string.Empty;

			foreach (string filePath in files)
			{
				int lineCount = CountLines(filePath);
				string relativePath = GetRelativePath(repoRoot, filePath);
				int limit = GetFileLimit(relativePath);

				if (lineCount > largest)
				{
					largest = lineCount;
					largestPath = relativePath;
				}

				if (lineCount > limit)
				{
					violations.Add((relativePath, lineCount, limit));
				}
			}

			Console.WriteLine();
			Console.WriteLine("=== File Compliance Summary ===");
			Console.WriteLine($"  Total .cs files analyzed: {files.Count}");
			Console.WriteLine($"  Default limit: {DefaultLineLimit} lines");
			Console.WriteLine($"  Exemptions: {ExplicitFileLimits.Count}");
			foreach ((string exemptPath, int exemptLimit) in ExplicitFileLimits.OrderBy(e => e.Key, StringComparer.Ordinal))
			{
				Console.WriteLine($"    {exemptPath}: ceiling {exemptLimit}");
			}

			Console.WriteLine($"  Largest file: {largestPath} ({largest} lines)");

			if (violations.Count > 0)
			{
				Console.WriteLine();
				Console.WriteLine($"  VIOLATIONS: {violations.Count}");
				foreach ((string path, int lines, int limit) in violations.OrderByDescending(v => v.Lines))
				{
					Console.WriteLine($"    {path}: {lines} lines (limit: {limit}, {lines - limit} over)");
				}
			}
			else
			{
				Console.WriteLine();
				Console.WriteLine("  All files comply with size limits!");
			}

			Console.WriteLine("===============================");

			// This test always passes - it's informational
			await Assert.That(files.Count).IsGreaterThan(0);
		}

		/// <summary>
		/// The exemption list is a ratchet, and a ratchet only works if it is kept honest:
		/// an entry whose file was since split (or renamed, or typo'd) exempts nothing and
		/// quietly raises the cap for a path that may exist again later.
		/// </summary>
		[Test]
		public async Task ExemptionsArePresentAndStillNeeded()
		{
			string repoRoot = ResolveRepoRoot();
			List<string> stale = new List<string>();

			foreach ((string relativePath, int limit) in ExplicitFileLimits)
			{
				string fullPath = Path.Combine(repoRoot, relativePath.Replace('/', Path.DirectorySeparatorChar));
				if (!File.Exists(fullPath))
				{
					stale.Add($"  {relativePath}: no such file - remove the entry (or fix the path; it must be repo-root-relative with forward slashes).");
					continue;
				}

				int lineCount = CountLines(fullPath);
				if (lineCount <= DefaultLineLimit)
				{
					stale.Add($"  {relativePath}: {lineCount} lines, now under the default {DefaultLineLimit} - remove the entry.");
				}
				else if (lineCount > limit)
				{
					// The size test reports this too; naming it here says which knob is wrong.
					stale.Add($"  {relativePath}: {lineCount} lines has grown past its {limit}-line ceiling - split the file, do not raise the ceiling.");
				}
			}

			await Assert.That(stale).IsEmpty()
				.Because("stale or outgrown ExplicitFileLimits entries:\n" + string.Join("\n", stale));
		}

		[Test]
		public async Task ScanExcludesBuildOutputDirectories()
		{
			string root = CreateScratchRoot();
			try
			{
				WriteSourceFile(Path.Combine(root, "Kept.cs"));
				WriteSourceFile(Path.Combine(root, "obj", "Debug", "net10.0", "Generated.cs"));
				WriteSourceFile(Path.Combine(root, "bin", "Debug", "net10.0", "Copied.cs"));
				WriteSourceFile(Path.Combine(root, "Nested", "AlsoKept.cs"));

				List<string> files = new List<string>();
				ScanDirectory(root, root, files);

				// Ordered comparison: TUnit's IsEquivalentTo defaults to CollectionOrdering.Any,
				// which would silently make this a set comparison.
				List<string> names = files.Select(f => Path.GetFileName(f)).OrderBy(n => n, StringComparer.Ordinal).ToList();
				await Assert.That(names).IsEquivalentTo(
					new List<string> { "AlsoKept.cs", "Kept.cs" },
					CollectionOrdering.Matching);
			}
			finally
			{
				Directory.Delete(root, true);
			}
		}

		[Test]
		public async Task ScanToleratesDirectoryDeletedMidWalk()
		{
			string root = CreateScratchRoot();
			try
			{
				// A directory can show up in the parent's listing and be gone before the scan
				// descends into it. Nothing is left in it to measure, so the walk continues.
				string vanished = Path.Combine(root, "VanishedMidWalk");
				List<string> files = new List<string>();

				ScanDirectory(root, vanished, files);

				await Assert.That(files.Count).IsEqualTo(0);
			}
			finally
			{
				Directory.Delete(root, true);
			}
		}

		private static string CreateScratchRoot()
		{
			string root = Path.Combine(Path.GetTempPath(), "ManifoldSharpFileCompliance", Path.GetRandomFileName());
			Directory.CreateDirectory(root);
			return root;
		}

		private static void WriteSourceFile(string filePath)
		{
			Directory.CreateDirectory(Path.GetDirectoryName(filePath)!);
			File.WriteAllText(filePath, "// scan fixture" + Environment.NewLine);
		}

		/// <summary>
		/// Count the lines in a file. See the file header for why this is the total and not
		/// MatterCAD's non-empty count.
		/// </summary>
		/// <remarks>
		/// <c>ReadAllLines</c> counts a final unterminated line, where <c>wc -l</c> (which
		/// counts newlines) does not — so this can read one higher than the shell on a file
		/// with no trailing newline. That is the more truthful "number of lines", and it is
		/// the direction that errs toward enforcing the cap rather than around it.
		/// </remarks>
		private static int CountLines(string filePath)
		{
			try
			{
				return File.ReadAllLines(filePath).Length;
			}
			catch (IOException)
			{
				return 0;
			}
			catch (UnauthorizedAccessException)
			{
				return 0;
			}
		}

		/// <summary>
		/// Get the line limit for a specific file path, relative to the repo root.
		/// </summary>
		private static int GetFileLimit(string relativePath)
		{
			string normalizedPath = relativePath.Replace('\\', '/');

			foreach ((string explicitPath, int explicitLimit) in ExplicitFileLimits)
			{
				string normalizedExplicit = explicitPath.Replace('\\', '/');
				if (normalizedPath.Equals(normalizedExplicit, StringComparison.OrdinalIgnoreCase)
					|| normalizedPath.EndsWith("/" + normalizedExplicit, StringComparison.OrdinalIgnoreCase))
				{
					return explicitLimit;
				}
			}

			return DefaultLineLimit;
		}

		/// <summary>
		/// Get all scanned source files under the three project directories.
		/// </summary>
		private static List<string> GetAllProjectFiles(string repoRoot)
		{
			List<string> files = new List<string>();
			foreach (string project in ScannedProjectDirectories)
			{
				string directory = Path.Combine(repoRoot, project);
				if (!Directory.Exists(directory))
				{
					// Loud, not silent: a renamed or moved project would otherwise quietly drop
					// out of the scan and the suite would still be green.
					throw new InvalidOperationException(
						$"Scanned project directory '{project}' was not found under '{repoRoot}'. "
						+ "If a project was renamed or moved, update ScannedProjectDirectories.");
				}

				ScanDirectory(repoRoot, directory, files);
			}

			files.Sort(StringComparer.Ordinal);
			return files;
		}

		private static void ScanDirectory(string repoRoot, string directory, List<string> files)
		{
			_ = repoRoot;

			if (ExcludedDirectories.Contains(Path.GetFileName(directory)))
			{
				return;
			}

			try
			{
				foreach (string file in Directory.GetFiles(directory))
				{
					if (IncludedExtensions.Contains(Path.GetExtension(file)))
					{
						files.Add(file);
					}
				}

				foreach (string subDir in Directory.GetDirectories(directory))
				{
					ScanDirectory(repoRoot, subDir, files);
				}
			}
			catch (UnauthorizedAccessException)
			{
				// Skip directories we can't access
			}
			catch (DirectoryNotFoundException)
			{
				// The directory was deleted between the parent's listing and this descent, so
				// there is nothing left in it to measure.
			}
		}

		/// <summary>
		/// Get a repo-root-relative path, always with forward slashes so the reported path and
		/// the <see cref="ExplicitFileLimits"/> keys are spelled the same on every OS.
		/// </summary>
		private static string GetRelativePath(string basePath, string fullPath)
		{
			return Path.GetRelativePath(basePath, fullPath).Replace('\\', '/');
		}

		/// <summary>
		/// Resolve the repository root.
		/// </summary>
		/// <remarks>
		/// Two independent routes, because each one alone has a failure mode. The compile-time
		/// <see cref="CallerFilePathAttribute"/> path is exact but baked into the assembly, so
		/// it is wrong for any run whose binaries were built somewhere else (a CI artifact
		/// rehydrated on another machine, a copied <c>bin/</c>). Walking up from the loaded
		/// assembly is location-truthful but only lands on a root that still contains the
		/// marker. Both are validated against <see cref="RepoMarkerFile"/>, and a run that
		/// satisfies neither throws rather than silently scanning the wrong tree.
		/// </remarks>
		private static string ResolveRepoRoot([CallerFilePath] string sourceFilePath = "")
		{
			// This file is at <repo>/ManifoldSharp.Tests/FileComplianceTests.cs — one level down.
			string? sourceDirectory = Path.GetDirectoryName(sourceFilePath);
			if (!string.IsNullOrEmpty(sourceDirectory))
			{
				string candidate = Path.GetFullPath(Path.Combine(sourceDirectory, ".."));
				if (File.Exists(Path.Combine(candidate, RepoMarkerFile)))
				{
					return candidate;
				}
			}

			DirectoryInfo? directory = new DirectoryInfo(AppContext.BaseDirectory);
			while (directory != null)
			{
				if (File.Exists(Path.Combine(directory.FullName, RepoMarkerFile)))
				{
					return directory.FullName;
				}

				directory = directory.Parent;
			}

			throw new InvalidOperationException(
				$"Could not locate the repo root: no '{RepoMarkerFile}' above the compile-time source path "
				+ $"('{sourceFilePath}') or above the test assembly ('{AppContext.BaseDirectory}').");
		}
	}
}
