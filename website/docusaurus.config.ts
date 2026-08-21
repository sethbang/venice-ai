import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Venice AI Python SDK',
  tagline: 'Production-ready Python SDK for Venice.ai',
  favicon: 'img/favicon.svg',

  url: 'https://venice-docs.sbang.dev',
  baseUrl: '/',

  organizationName: 'sethbang',
  projectName: 'venice-py',

  onBrokenLinks: 'throw',

  future: {
    v4: true,
    faster: true,
  },

  // '.md' files (migrated guides + pydoc-markdown output) parse as lenient
  // CommonMark; '.mdx' stays strict MDX.
  markdown: {
    format: 'detect',
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/sethbang/venice-py/tree/main/website/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  plugins: [
    [
      'docusaurus-plugin-llms',
      {
        generateLLMsTxt: true,
        generateLLMsFullTxt: true,
        generateMarkdownFiles: true,
        title: 'Venice AI Python SDK Documentation',
        description: 'Complete API reference and guides for the Venice AI Python SDK (v2+).',
        includeOrder: ['getting-started', 'guides/**', 'api/**'],
        excludeImports: true,
        removeDuplicateHeadings: true,
        customLLMFiles: [
          {
            filename: 'llms-api.txt',
            title: 'Venice AI Python SDK — API Reference',
            includePatterns: ['api/**'],
            fullContent: true,
          },
        ],
      },
    ],
  ],

  themeConfig: {
    navbar: {
      title: 'Venice AI Python SDK',
      items: [
        {type: 'docSidebar', sidebarId: 'guidesSidebar', position: 'left', label: 'Docs'},
        {type: 'docSidebar', sidebarId: 'apiSidebar', position: 'left', label: 'API Reference'},
        {href: 'https://github.com/sethbang/venice-py', label: 'GitHub', position: 'right'},
      ],
    },
    footer: {
      style: 'dark',
      copyright:
        'Unofficial, community-maintained SDK — not affiliated with or endorsed by Venice AI. Official resources: <a href="https://venice.ai/">Venice.ai</a>.',
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
