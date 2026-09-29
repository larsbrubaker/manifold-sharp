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

// AssertionOrderingAdaptationTests.cs — NOT A PORT. A C#-only compliance gate
// with no counterpart in manifold-rust, counted with the adaptation tests and
// never toward the test-for-test tally. It makes CLAUDE.md's "Ordered
// assertions" translation rule executable.
//
// Rust's assert_eq! on a Vec or slice checks order. TUnit's IsEquivalentTo /
// IsNotEquivalentTo on a collection does not unless given
// CollectionOrdering.Matching: its default, CollectionOrdering.Any, silently
// turns a ported sequence comparison into a set comparison, so a triangle list
// or vertex row in the wrong order would pass. The unordered match is also
// quadratic over repeated values, which a large index list is full of. So every
// equivalence compare in this repo's tests has to say which it means: Matching
// for a sequence (every ported assert_eq!, including the ones where the Rust
// sorts first and the port sorts the same way), Any for a genuine set such as a
// HashSet or a directory scan. Adapted from agg-sharp's
// Tests/Agg.Tests/Other/PixelBufferAssertionTests.EveryEquivalenceStatesItsOrdering.

using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class AssertionOrderingAdaptationTests
	{
		/// <summary>
		/// The file that marks the repository root.
		/// </summary>
		private const string RepoMarkerFile = "ManifoldSharp.sln";

		[Test]
		public async Task EveryEquivalenceStatesItsOrdering()
		{
			List<(string Location, string Text)> calls = EquivalenceCalls().ToList();
			List<string> unstated = calls
				.Where(call => !call.Text.Contains("CollectionOrdering."))
				.Select(call => call.Location)
				.ToList();

			// A scan that found nothing would pass vacuously if the root or the pattern broke.
			await Assert.That(calls.Count).IsGreaterThan(0);
			await Assert.That(unstated).IsEmpty()
				.Because("pass CollectionOrdering.Matching for a sequence (a ported assert_eq! on a Vec) or "
					+ "CollectionOrdering.Any for a genuine set: " + string.Join(", ", unstated));
		}

		/// <summary>
		/// Every IsEquivalentTo( / IsNotEquivalentTo( call in the repo's C# sources, with its
		/// whole argument list (which may run over several lines) and the receiver on the
		/// same line. bin/ and obj/ are skipped, and so is this file: its own comments and
		/// string literals name the calls.
		/// </summary>
		private static IEnumerable<(string Location, string Text)> EquivalenceCalls(
			[CallerFilePath] string thisFile = "")
		{
			string repoRoot = ResolveRepoRoot(thisFile);
			foreach (string file in Directory.EnumerateFiles(repoRoot, "*.cs", SearchOption.AllDirectories))
			{
				string relative = Path.GetRelativePath(repoRoot, file).Replace('\\', '/');
				string[] parts = relative.Split('/');
				if (parts.Contains("bin") || parts.Contains("obj") || parts.Contains(".git")
					|| Path.GetFullPath(file) == Path.GetFullPath(thisFile))
				{
					continue;
				}

				string text = File.ReadAllText(file);
				foreach (Match match in Regex.Matches(text, @"Is(Not)?EquivalentTo\("))
				{
					int close = MatchingParen(text, match.Index + match.Length);
					int lineStart = text.LastIndexOf('\n', match.Index) + 1;
					int line = 1 + text.Take(match.Index).Count(c => c == '\n');
					yield return ($"{relative}:{line}", text.Substring(lineStart, close - lineStart));
				}
			}
		}

		/// <summary>
		/// The index of the ')' closing a call whose argument list starts at
		/// <paramref name="start"/>, skipping parentheses inside string and char literals.
		/// </summary>
		private static int MatchingParen(string text, int start)
		{
			int depth = 1;
			char quote = '\0';
			for (int i = start; i < text.Length; i++)
			{
				char c = text[i];
				if (quote != '\0')
				{
					if (c == '\\')
					{
						i++;
					}
					else if (c == quote)
					{
						quote = '\0';
					}
				}
				else if (c == '"' || c == '\'')
				{
					quote = c;
				}
				else if (c == '(')
				{
					depth++;
				}
				else if (c == ')' && --depth == 0)
				{
					return i;
				}
			}

			return text.Length;
		}

		/// <summary>
		/// The repository root: one level above this file at compile time, or failing that
		/// (binaries built elsewhere) the first ancestor of the test assembly holding the
		/// marker. The same two routes as FileComplianceTests.ResolveRepoRoot, for the same
		/// reasons; a run that finds neither throws rather than scanning the wrong tree.
		/// </summary>
		private static string ResolveRepoRoot(string sourceFilePath)
		{
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
				$"Could not locate the repo root: no '{RepoMarkerFile}' above '{sourceFilePath}' "
				+ $"or above the test assembly ('{AppContext.BaseDirectory}').");
		}
	}
}
