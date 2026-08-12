import {Redirect} from '@docusaurus/router';

// This is a single-SDK documentation site, so the root sends visitors straight
// to the docs rather than a standalone splash page.
export default function Home(): JSX.Element {
  return <Redirect to="/docs/getting-started" />;
}
